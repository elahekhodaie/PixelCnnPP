import torch
from torchvision import datasets

# data I/O
use_arg_parser = False  # whether or not to use arg_parser
data_dir = 'data'  # Location for the dataset
save_dir = 'models'  # Location for parameter checkpoints and samples
dataset = datasets.MNIST
normal_classes = [8]
test_classes = [1, 3, 8]
print_every = 1  # how many iterations between print statements
save_interval = 10  # Every how many epochs to write checkpoint/samples?
load_params = None  # Restore training from previous model checkpoint?

# data loader
batch_size = 64  # Batch size during training per GPU
dataloader_num_workers = 1
dataloader_pin_memory = True
dataloader_shuffle = True
dataloader_drop_last = True

# device
use_tpu = False
num_cores = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if use_tpu:
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

# model
nr_resnet = 5  # Number of residual blocks per stage of the model
nr_filters = 160  # number of filters to use across the model. (Higher = larger model)
nr_logistic_mix = 5  # Number of logistic components in the mixture. (Higher = more flexible model)
lr = 0.0002  # Base learning rate
lr_decay = 0.999995  # Learning rate decay, applied every step of the optimization
max_epochs = 5000  # How many epochs to run in total

# samples
sample_batch_size = 25

# Reproducability
seed = 1  # Random seed to use
