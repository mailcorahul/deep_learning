import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np

class MNISTDataset(Dataset):

    def __init__(self):
        """
        Downloads MNIST dataset for training, and also loads test data from disk.
            train_data: tensor containing train images
            train_labels: tensor containing train labels
        """

        TEST_RATIO = 0.05
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        idxs = np.arange(mnist_trainset.train_data.size(0))
        np.random.shuffle(idxs)

        self.train_data = torch.reshape(
                mnist_trainset.train_data[idxs].float(), (-1,1,28,28))/255.
        self.data_size = self.train_data.size(0)

        idx = int(self.data_size*TEST_RATIO)
        self.test_data = self.train_data[:idx]
        self.train_data = self.train_data[idx:]
        self.train_len = self.train_data.size(0)

        print('Train images -- {}'.format(self.train_data.size()))
        print('Test images -- {}'.format(self.test_data.size()))

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):

        input = self.train_data[idx]
        return input, input