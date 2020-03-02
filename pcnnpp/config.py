# data I/O
use_arg_parser = False # whether or not to use arg_parser
data_dir = 'data'  # Location for the dataset
save_dir = 'models'  # Location for parameter checkpoints and samples
dataset = 'mnist'  # Can be either cifar|mnist
print_every = 50  # how many iterations between print statements
save_interval = 10  # Every how many epochs to write checkpoint/samples?
load_params = None  # Restore training from previous model checkpoint?

# model
nr_resnet = 5  # Number of residual blocks per stage of the model
nr_filters = 160  # number of filters to use across the model. (Higher = larger model)
nr_logistic_mix = 10  # Number of logistic components in the mixture. (Higher = more flexible model)
lr = 0.0002  # Base learning rate
lr_decay = 0.999995  # Learning rate decay, applied every step of the optimization
batch_size = 64  # Batch size during training per GPU
max_epochs = 5000  # How many epochs to run in total

# Reproducability
seed = 1  # Random seed to use

