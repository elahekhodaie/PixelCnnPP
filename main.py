import time
import os
import torch.utils.data
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from pcnnpp.model import *
from pcnnpp import config
if config.use_arg_parser:
    import pcnnpp.utils.argparser
    config = pcnnpp.utils.argparser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# reproducibility
torch.manual_seed(config.seed)
np.random.seed(config.seed)

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(config.lr, config.nr_resnet, config.nr_filters)
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

sample_batch_size = 25
obs = (1, 28, 28) if 'mnist' in config.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling = lambda x: (x - .5) * 2.
rescaling_inv = lambda x: .5 * x + .5
kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

if 'mnist' in config.dataset:
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(config.data_dir, download=True,
                                                              train=True, transform=ds_transforms),
                                               batch_size=config.batch_size,
                                               shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(config.data_dir, train=False,
                                                             transform=ds_transforms), batch_size=config.batch_size,
                                              shuffle=True, **kwargs)

    loss_op = lambda real, fake: discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x: sample_from_discretized_mix_logistic_1d(x, config.nr_logistic_mix)

elif 'cifar' in config.dataset:
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(config.data_dir, train=True,
                                                                download=True, transform=ds_transforms),
                                               batch_size=config.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(config.data_dir, train=False,
                                                               transform=ds_transforms), batch_size=config.batch_size,
                                              shuffle=True, **kwargs)

    loss_op = lambda real, fake: discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x: sample_from_discretized_mix_logistic(x, config.nr_logistic_mix)
else:
    raise Exception('{} dataset not in [mnist, cifar10]'.format(config.dataset))

model = PixelCNN(nr_resnet=config.nr_resnet, nr_filters=config.nr_filters,
                 input_channels=input_channels, nr_logistic_mix=config.nr_logistic_mix)
model = model.to(device)

if config.load_params:
    load_part_of_model(model, config.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')

optimizer = optim.Adam(model.parameters(), lr=config.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)


def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.to(device)
    for i in range(obs[1]):
        for j in range(obs[2]):
            data_v = Variable(data, volatile=True)
            out = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data


print('starting training')
writes = 0
for epoch in range(config.max_epochs):
    model.train(True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
        input = Variable(input)
        output = model(input)
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.data)
        train_loss += loss.data
        if (batch_idx + 1) % config.print_every == 0:
            deno = config.print_every * config.batch_size * np.prod(obs) * np.log(2.)
            writer.add_scalar('train/bpd', (train_loss / deno), writes)
            print('loss : {:.4f}, time : {:.4f}'.format(
                (train_loss / deno),
                (time.time() - time_)))
            train_loss = 0.
            writes += 1
            time_ = time.time()

    # decrease learning rate
    scheduler.step()

    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    for batch_idx, (input, _) in enumerate(test_loader):
        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
        input.requires_grad = True
        output = model(input)
        loss = loss_op(input, output)
        test_loss += loss.data[0]
        del loss, output

    deno = batch_idx * config.batch_size * np.prod(obs) * np.log(2.)
    writer.add_scalar('test/bpd', (test_loss / deno), writes)
    print('test loss : %s' % (test_loss / deno))

    if (epoch + 1) % config.save_interval == 0:
        torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
        print('sampling...')
        sample_t = sample(model)
        sample_t = rescaling_inv(sample_t)
        utils.save_image(sample_t, 'images/{}_{}.png'.format(model_name, epoch),
                         nrow=5, padding=0)
