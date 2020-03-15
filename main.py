import time
import os
import torch
import numpy as np
from torch.optim import lr_scheduler, Adam
from torchvision import utils
from tensorboardX import SummaryWriter
from pcnnpp.model import init_model, PixelCNN
from pcnnpp import config
from pcnnpp.data import DatasetSelection, rescaling_inv
from pcnnpp.utils.functions import get_loss_function
from pcnnpp.utils.evaluation import sample, plot_loss, evaluate, plot_evaluation, show_extreme_cases
from torch import optim
import matplotlib.pyplot as plt

if config.use_tpu:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.utils.utils as xu

if config.use_arg_parser:
    import pcnnpp.utils.argparser

    config = pcnnpp.utils.argparser.parse_args()

# reproducibility
torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(config.seed)
np.random.seed(config.seed)

model = None
dataset_train = None
dataset_validation = None


def train():
    global model
    validation_losses = []
    train_losses = []
    print('starting training')
    # starting up data loaders
    print("loading training data")
    dataset_train = DatasetSelection(train=True, classes=config.normal_classes)
    print('loading validation data')
    dataset_validation = DatasetSelection(train=False, classes=config.normal_classes)
    print('loading test data')
    dataset_test = DatasetSelection(train=False, classes=config.test_classes)

    train_sampler = None
    validation_sampler = None
    test_sampler = None
    if config.use_tpu:
        print('creating tpu sampler')
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_validation,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False
        )
        print('tpu samplers created')
    train_loader = dataset_train.get_dataloader(sampler=train_sampler, shuffle=not config.use_tpu)
    validation_loader = dataset_validation.get_dataloader(sampler=validation_sampler, shuffle=not config.use_tpu, )
    test_loader = dataset_test.get_dataloader(sampler=test_sampler, shuffle=False, )

    input_shape = dataset_validation.input_shape()
    loss_function = get_loss_function(input_shape)

    # setting up tensorboard data summerizer
    writer = SummaryWriter(log_dir=os.path.join(config.log_dir, config.model_name))

    # initializing model
    model = init_model(input_shape)

    print("initializing optimizer & scheduler")

    # change the optimizer to non parameter mode for adversarial  mode

    #optimizer = Adam(model.parameters(), lr=config.lr)
    optimizer = optim.SGD(model.parameters(), lr = config.lr)
   # optimizer = optim.SGD()

    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=config.lr_multiplicative_factor_lambda,
                                      last_epoch=config.start_epoch - 1)

    def train_loop(data_loader, writes=0):
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=config.device)
        train_loss = 0.
        last_train_loss = 0.
        new_writes = 0
        time_ = time.time()
        if config.use_tpu:
            tracker = xm.RateTracker()
        model.train()
        for batch_idx, (input, _) in enumerate(data_loader):
            input = input.to(config.device, non_blocking=True)
            if config.noising_factor is not None:
                false_input = input + config.noising_factor * config.noise_function(input.shape)
                false_input.clamp_(min=-1, max=1)
                output = model(false_input)
            elif config.adversarial_training_mode is True:

                for epoch in range(60):
                    #start = datetime.now()
                    itr = 0
                    for x, y in train_loader:
                        x_adv = pgd_attack(optimizer,model, x, y, loss_function, iteration_num= 40, step_size=0.01,
                                           eps=0.2, eps_norm='inf', step_norm='inf')

                        adv_img = Variable(x_adv).cuda()

                        #------------------------backward---------------------------------

                        predicted_y = model(x_adv)
                        loss= loss_function(predicted_y, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        itr += 1
                        train_loss += loss.item()
                    #------------------------print results ---------------------------------
                   # print('epoch [{}/{}], loss:{:.4f}.format(epoch + 1, num_epochs, total_loss / itr))
                        #for every 10 epochs the results are printed
                        if epoch % 10 == 0:
                            raw_input = to_img(x.cpu().data)
                            output_pic = to_img(y.cpu().data)
                            adv_input = to_img(adv_img.cpu().data)
                            show_process (raw_input, output_pic, adv_input, train=True, attack=True)
                         #   print(\"loss_latent : \", latent_loss.item())
                         #   print(\"loss_AE : \", AE_loss.item())
                        #if epoch % 10 == 0:
                           # torch.save({
                            #    'epoch': epoch + last_epoch,
                            #   'model_state_dict': model.state_dict(),
                            #    'optimizer_state_dict': optimizer.state_dict(),
                            #    'loss': loss,
                            #    }, './CIFAR_64_32_random_eps=0.1_0_latent=64_k=0.1.pth')
                        print(datetime.now() - start)

            else:
                #optimizer = Adam(model.parameters(), lr=config.lr)
                output = model(input)
                loss = loss_function(input, output)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss
            # if config.use_tpu:
            #     xm.optimizer_step(optimizer)
            #     tracker.add(config.batch_size)
            # else:
                optimizer.step()

            if config.print_every and (batch_idx + 1) % config.print_every == 0 :
                deno = config.print_every * config.batch_size * np.prod(input_shape) * np.log(2.)
                if not config.use_tpu:
                    writer.add_scalar('train/bpd', (train_loss / deno), writes + new_writes)

                print('\t{:3d}/{:3d} - loss : {:.4f}, time : {:.3f}s'.format(
                    batch_idx // config.print_every + 1,
                    len(train_loader) // config.print_every,
                    (train_loss / deno),
                    (time.time() - time_)
                ))
                last_train_loss = train_loss
                train_loss = 0.
                new_writes += 1
                time_ = time.time()
            del input, _, loss, output

        return new_writes, (last_train_loss / deno)

    def validation_loop(data_loader, writes=0):
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=config.device)
        model.eval()
        test_loss = 0.
        with torch.no_grad():
            for batch_idx, (input, _) in enumerate(data_loader):
                input = input.to(config.device, non_blocking=True)
                output = model(input)
                loss = loss_function(input, output)
                test_loss += loss
                del loss, output

            deno = batch_idx * config.batch_size * np.prod(input_shape) * np.log(2.)
            writer.add_scalar('validation/bpd', (test_loss / deno), writes)
            print('\t{}epoch {:4} validation loss : {:.4f}'.format(
                '' if not config.use_tpu else xm.get_ordinal(),
                epoch,
                (test_loss / deno)
            ),
                flush=True
            )

            if config.save_interval and (epoch + 1) % config.save_interval == 0:
                torch.save(model.state_dict(), config.models_dir + '/{}_{}.pth'.format(config.model_name, epoch))
                print('\tsampling epoch {:4}'.format(
                    epoch
                ))
                sample_t = sample(model, input_shape)
                sample_t = rescaling_inv(sample_t)
                utils.save_image(sample_t, config.samples_dir + '/{}_{}.png'.format(config.model_name, epoch),
                                 nrow=5, padding=0)
            return test_loss / deno

    try:
        writes = 0
        for epoch in range(config.start_epoch, config.max_epochs):
            print('epoch {:4} - lr: {}'.format(epoch, optimizer.param_groups[0]["lr"]))
            if config.use_tpu:
                para_loader = pl.ParallelLoader(train_loader, [config.device])
                train_loop(para_loader.per_device_loader(config.device), writes)
                xm.master_print("\tFinished training epoch {}".format(epoch))
            else:
                new_writes, train_loss = train_loop(train_loader, writes)
                train_losses.append(train_loss)
                writes += new_writes

            # learning rate schedule
            scheduler.step(epoch)

            if config.use_tpu:
                para_loader = pl.ParallelLoader(validation_loader, [config.device])
                validation_loop(para_loader.per_device_loader(config.device), writes)
            else:
                validation_loss = validation_loop(validation_loader, writes)
                validation_losses.append(validation_loss)
                model_name = f'{"DCNNpp" if config.noising_factor is not None else "PCNNpp"}-E{epoch}'
                # evaluation and loss tracking
                if config.plot_every and (epoch + 1) % config.plot_every == 0:
                    plot_loss(
                        train_losses,
                        validation_losses,
                        model_name=f'{"DCNNpp" if config.noising_factor is not None else "PCNNpp"}-{optimizer.param_groups[0]["lr"]:.7f}'
                        , save_path=config.losses_dir + f'/Losses{model_name}.png',
                    )

                if config.evaluate_every and (epoch + 1) % config.evaluate_every == 0:
                    eval_data = evaluate(model, dataset_test, test_loader)
                    plot_evaluation(
                        eval_data,
                        model_name=f'{"DCNNpp" if config.noising_factor is not None else "PCNNpp"}-E{epoch}',
                        save_path=config.evaluation_dir + f'/EvalPlot{model_name}.png'
                    )
                    show_extreme_cases(
                        eval_data,
                        model_name=model_name,
                        save_dir=config.extreme_cases_dir
                    )

            writes += 1
    except KeyboardInterrupt:
        pass
    return model, train_losses, validation_losses

def pgd_attack(optimizer,model, x, y, loss_function, iteration_num , step_size, step_norm, eps, eps_norm,
                               clamp=(-1,1), y_target=None):
    print("has reached this pgd attack")
    #inputs = inputs.cuda()

    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None

    num_channels = x.shape[0]
    original_img = x


    for i in range(iteration_num):

        # x_adv is the adversarial built model which is given to calculate the loss as an input
        _x_adv = x_adv.clone().detach().requires_grad_(True)
        prediction_output = model(_x_adv)


# have to add the model.grad requires  false to turn off other parameters in grad

        loss = loss_function(prediction_output, y_target if targeted else y)
        optimizer.zero_grad()
        loss.backward()

       # step_size = alpha
        with torch.no_grad():
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step_size
            else:
                gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1) \
                    .norm(step_norm, dim=-1) \
                    .view(-1, num_channels, 1, 1)

            if targeted:
                # in the non training state with incorrect labels
                x_adv -= gradients
            else:
                # the model parameters
                x_adv += gradients

        # Project back
        #this part calculates the norm and finds the similarity between the constructed image
        # and the first input image , it has two different modes L2 mode or L inf mode
        if eps_norm == 'inf':
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        else:
            delta = x_adv - x
            # first dimension is the batch dimension
            similarity_value = delta.view(delta.shape[0], -1).norm(norm, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(norm, dim=1)
            scaling_factor[similarity_value] = eps
            delta *= eps / scaling_factor.view(-1, 1, 1, 1)
            x_adv = x + delta

        #the clamp is between -1, 1
        x_adv = x_adv.clamp(*clamp)

    return x_adv.detach()






def show_process (input_img, recons_img, attacked_img = None, train=True, attack=False):
    n = input_img.shape[0]
    if train:
        print("Inputs:")
        show(input_img[0:n].view((1,-1,28,28))[0].cpu())
        # Calculate reconstructions\n",
        if attack:
            print("Inputs after attack:")
            show(attacked_img[0:n].view((1, -1, 28, 28))[0].cpu().detach().numpy())
        print("Reconstructions:")
        show(recons_img[0:n].view((1,-1,28,28))[0].cpu().detach().numpy())
    else:
        print("Test Inputs:")
        show(input_img[0:n].view((1, -1, 28, 28))[0].cpu())
        # Calculate reconstructions\n",
        print("Test Reconstructions:")
        show(recons_img[0:n].view((1, -1, 28, 28))[0].cpu().detach().numpy())


def show(image_batch, rows=1):
        # Set Plot dimensions\n",
        cols = np.ceil(image_batch.shape[0] / rows)
        plt.rcParams['figure.figsize'] = (0.0 + cols, 0.0 + rows) # set default size of plots\n",
        for i in range(image_batch.shape[0]):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image_batch[i], cmap=gray)
            plt.axis('off')
        plt.show()





if config.use_tpu:
    def train_on_tpu():
        def trainer(rank, CONFIG):
            global config
            config = CONFIG
            config.device = xm.xla_device()
            torch.set_default_tensor_type('torch.FloatTensor')
            train()

        xmp.spawn(trainer, args=(config,), nprocs=config.num_cores,
                  start_method='fork')

if config.train:
    if config.use_tpu:
        train_on_tpu()
    else:
        train()
