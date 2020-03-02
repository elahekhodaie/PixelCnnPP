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

if config.use_arg_parser:
    import pcnnpp.utils.argparser

    config = pcnnpp.utils.argparser.parse_args()

dataset_train = DatasetSelection(train=True, classes=config.normal_classes)
dataset_test = DatasetSelection(train=True, classes=config.test_classes)
train_loader = dataset_train.get_dataloader()
test_loader = dataset_test.get_dataloader()

input_shape = dataset_test.input_shape()
print(input_shape)
loss_function = get_loss_function(input_shape)

# reproducibility
torch.manual_seed(config.seed)
np.random.seed(config.seed)

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(config.lr, config.nr_resnet, config.nr_filters)
# assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

model = PixelCNN(nr_resnet=config.nr_resnet, nr_filters=config.nr_filters,
                 input_channels=input_shape[0], nr_logistic_mix=config.nr_logistic_mix)
model = model.to(config.device)

if config.load_params:
    load_part_of_model(model, config.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')

optimizer = Adam(model.parameters(), lr=config.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)

print('starting training')
writes = 0
for epoch in range(config.max_epochs):
    model.train(True)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=config.device)
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input, _) in enumerate(train_loader):
        input = input.to(config.device, non_blocking=True)
        input.requires_grad = True
        output = model(input)
        loss = loss_function(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
        if (batch_idx + 1) % config.print_every == 0:
            deno = config.print_every * config.batch_size * np.prod(input_shape) * np.log(2.)
            writer.add_scalar('train/bpd', (train_loss / deno), writes)
            print('{:4d}/{:4d}- {:4d} - loss : {:.4f}, time : {:.4f}'.format(batch_idx, len(train_loader), epoch,
                                                                             (train_loss / deno),
                                                                             (time.time() - time_)))
            train_loss = 0.
            writes += 1
            time_ = time.time()

    # decrease learning rate
    scheduler.step(epoch)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device=config.device)
    model.eval()
    test_loss = 0.
    for batch_idx, (input, _) in enumerate(test_loader):
        input = input.to(config.device, non_blocking=True)
        input.requires_grad = True
        output = model(input)
        loss = loss_function(input, output)
        test_loss += loss.data[0]
        del loss, output

    deno = batch_idx * config.batch_size * np.prod(input_shape) * np.log(2.)
    writer.add_scalar('test/bpd', (test_loss / deno), writes)
    print('test loss : %s' % (test_loss / deno))

    if (epoch + 1) % config.save_interval == 0:
        torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
        print('sampling...')
        sample_t = sample(model, input_shape)
        sample_t = rescaling_inv(sample_t)
        utils.save_image(sample_t, 'images/{}_{}.png'.format(model_name, epoch),
                         nrow=5, padding=0)
