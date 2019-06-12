import torch
from torch.autograd import Variable
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

        #print(torch.min(mnist_trainset.train_labels), torch.max(mnist_trainset.train_labels))
        #print(mnist_trainset.train_labels.size())
        
        # reshape input data to (1, 784) and normalize to range [0., 1.]
        self.train_data = torch.reshape(
                mnist_trainset.train_data[idxs].float(), (-1,1,28,28))/255.
        self.data_size = self.train_data.size(0)
        self.train_len = self.train_data.size(0)
        self.train_label = torch.Tensor([1]).long() # since there is only one class - 'real' image

        print('Train images -- {}'.format(self.train_data.size()))

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):

        input = self.train_data[idx]
        return input, self.train_label