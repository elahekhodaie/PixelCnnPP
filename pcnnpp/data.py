from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pcnnpp.config as config

rescaling = lambda x: (x - .5) * 2.
rescaling_inv = lambda x: .5 * x + .5
default_transform = transforms.Compose([transforms.ToTensor(), rescaling])
kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}


class DatasetSelection(Dataset):
    def __init__(self,
                 dataset=config.dataset,
                 root=config.data_dir,
                 train=True,
                 classes=tuple(range(0, 9)),
                 transform=default_transform,
                 target_transform=lambda x: x,
                 download=False,
                 ):
        self.whole_data = dataset(root, train, transform=transform, target_transform=target_transform,
                                  download=download)
        self.data = [point for point in self.whole_data if point[1] in classes]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def input_shape(self):
        return self.data[0][0].shape

    def get_dataloader(self, batch_size=config.batch_size,
                       shuffle=config.dataloader_shuffle,
                       num_workers=config.dataloader_num_workers,
                       pin_memory=config.dataloader_pin_memory,
                       drop_last=config.dataloader_drop_last,
                       ):
        return DataLoader(self, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory,
                          drop_last=drop_last)
