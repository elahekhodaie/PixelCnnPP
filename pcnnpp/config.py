import torch
from torchvision import datasets
from pathlib import Path

# run on import
train = False

# data I/O
output_root = ''
use_arg_parser = False  # whether or not to use arg_parser
data_dir = output_root + 'data'  # Location for the dataset
models_dir = output_root + 'models'  # Location for parameter checkpoints and samples
samples_dir = output_root + 'samples'
log_dir = output_root + 'log'

train_dataset = datasets.MNIST
normal_classes = [8]

test_classes = list(range(0, 10))
test_dataset = datasets.MNIST

# log and save config
print_every = 11  # how many iterations between print statements
evaluate_print_every = 20 # how many iterations between print statements in evalutation
save_interval = 64  # Every how many epochs to write checkpoint/samples?
plot_every = 32  # plot loss epochs interval
evaluate_every = 64  # evaluation interval
load_params = None  # Restore training from previous model checkpoint (specify the model dump file path)

start_epoch = 0

# data loader
batch_size = 256  # Batch size during training per GPU
test_batch_size = batch_size
dataloader_num_workers = 4
dataloader_pin_memory = True
dataloader_shuffle = True
dataloader_drop_last = True

# device
use_tpu = False
num_cores = 8
if not use_tpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model & training parameters
nr_resnet = 4  # Number of residual blocks per stage of the model
nr_filters = 20  # number of filters to use across the model. (Higher = larger model)
nr_logistic_mix = 3  # Number of logistic components in the mixture. (Higher = more flexible model)
lr_decay = 0.999995  # Learning rate decay, applied every step of the optimization
lr_half_schedule = 512  # interval of epochs to reduce learning rate 50%
lr_multiplicative_factor_lambda = lambda epoch: 0.5 if (epoch + 1) % lr_half_schedule else lr_decay
lr = 0.0002 * lr_decay ** start_epoch  # Base learning rate
noising_factor = 0.2  # the noise to add to each input while training the model
noise_function = lambda x: 2 * torch.FloatTensor(*x).to(device).uniform_() - 1  # (x will be the input shape tuple)
max_epochs = 4096  # How many epochs to run in total
model_name = '{}lr{:.5f}resnet{}filter{}nrmix{}'.format(
    f'd{noising_factor}pcnnpp' if noising_factor is not None else 'pcnnpp', lr, nr_resnet, nr_filters,
    nr_logistic_mix)

# samples
sample_batch_size = 25

# Reproducability
seed = 1  # Random seed to use

# ensuring the existance of output directories
Path(output_root).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(models_dir).mkdir(parents=True, exist_ok=True)
Path(samples_dir).mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)
