import time
import os
import torch
import numpy as np
from torch.optim import lr_scheduler, Adam
from torchvision import utils
from tensorboardX import SummaryWriter
from pcnnpp.model import PixelCNN, load_part_of_model
from pcnnpp import config
from pcnnpp.data import DatasetSelection, rescaling_inv
from pcnnpp.utils.functions import sample, get_loss_function

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

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(config.lr, config.nr_resnet, config.nr_filters)


def train():
    print('starting training')
    # starting up data loaders
    print("loading data")
    dataset_train = DatasetSelection(train=True, classes=config.normal_classes)
    dataset_test = DatasetSelection(train=True, classes=config.test_classes)

    train_sampler = None
    if config.use_tpu:
        print('creating tpu sampler')
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)
        print('tpu sampler created')

    input_shape = dataset_test.input_shape()
    loss_function = get_loss_function(input_shape)

    test_loader = dataset_test.get_dataloader(shuffle=False)
    train_loader = dataset_train.get_dataloader(sampler=train_sampler)

    # setting up tensorboard data summerizer
    writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

    # initializing model
    print("initializing model")
    model = PixelCNN(nr_resnet=config.nr_resnet, nr_filters=config.nr_filters,
                     input_channels=input_shape[0], nr_logistic_mix=config.nr_logistic_mix)
    model = model.to(config.device)

    if config.load_params:
        load_part_of_model(model, config.load_params)
        # model.load_state_dict(torch.load(config.load_params))
        print('model parameters loaded')

    print("initializing optimizer & scheduler")
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
    writes = 0

    def train_loop(data_loader):
        global time
        global writes

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=config.device)
        train_loss = 0.
        time_ = time.time()
        if config.use_tpu:
            tracker = xm.RateTracker()
        model.train()
        for batch_idx, (input, _) in enumerate(data_loader):
            input = input.to(config.device, non_blocking=True)
            output = model(input)
            loss = loss_function(input, output)
            optimizer.zero_grad()
            loss.backward()
            if config.use_tpu:
                xm.optimizer_step(optimizer)
                tracker.add(config.batch_size)
            else:
                optimizer.step()
            train_loss += loss
            if (batch_idx + 1) % config.print_every == 0:
                deno = config.print_every * config.batch_size * np.prod(input_shape) * np.log(2.)
                if not config.use_tpu:
                    writer.add_scalar('train/bpd', (train_loss / deno), writes)
                    writes += 1

                print('{:4d}/{:4d}- {:4d} - loss : {:.4f}, time : {:.4f}'.format(batch_idx, len(train_loader), epoch,
                                                                                 (train_loss / deno),
                                                                                 (time.time() - time_)))
                train_loss = 0.
                time = time.time()
        del loss, output

    def test_loop(data_loader):

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=config.device)
        model.eval()
        test_loss = 0.
        for batch_idx, (input, _) in enumerate(data_loader):
            input = input.to(config.device, non_blocking=True)
            output = model(input)
            loss = loss_function(input, output)
            test_loss += loss.data[0]
            del loss, output

        deno = batch_idx * config.batch_size * np.prod(input_shape) * np.log(2.)
        if not config.use_tpu:
            writer.add_scalar('test/bpd', (test_loss / deno), writes)

        print('{}-epoch {:04}test loss : %s'.format(None if not config.use_tpu else xm.get_ordinal(), epoch,
                                                    (test_loss / deno)), flush=True)

        if (epoch + 1) % config.save_interval == 0:
            torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
            print('sampling...')
            sample_t = sample(model, input_shape)
            sample_t = rescaling_inv(sample_t)
            utils.save_image(sample_t, 'images/{}_{}.png'.format(model_name, epoch),
                             nrow=5, padding=0)

    for epoch in range(config.max_epochs):
        if config.use_tpu:
            para_loader = pl.ParallelLoader(train_loader, [config.device])
            train_loop(para_loader.per_device_loader(config.device))
            xm.master_print("Finished training epoch {}".format(epoch))
        else:
            train_loop(train_loader)
            print("Finished training epoch {}".format(epoch))
        scheduler.step(epoch)
        if config.use_tpu:
            para_loader = pl.ParallelLoader(test_loader, [config.device])
            test_loop(para_loader.per_device_loader(config.device))
        else:
            test_loop(test_loader)


if config.use_tpu:
    def trainer(rank):
        train()


    xmp.spawn(trainer, nprocs=config.num_cores,
              start_method='fork')
else:
    train()
