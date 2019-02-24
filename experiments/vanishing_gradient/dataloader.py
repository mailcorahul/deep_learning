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

        TEST_RATIO = 0.05;
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        idxs = np.arange(mnist_trainset.train_data.size(0))
        np.random.shuffle(idxs);

        self.train_data = mnist_trainset.train_data[idxs];
        self.train_labels = mnist_trainset.train_labels[idxs];
        self.data_size = self.train_data.size(0);

        idx = int(self.data_size*TEST_RATIO);
        self.test_data = self.train_data[:idx];
        self.test_labels = self.train_labels[:idx];
        self.train_data = self.train_data[idx:];
        self.train_labels = self.train_labels[idx:];
        self.train_len = self.train_labels.size(0);

        print('Train images {}, Train labels {}'.format(self.train_data.size(),
            self.train_labels.size()));
        print('Test images {}, Test labels {}'.format(self.test_data.size(),
            self.test_labels.size()));

    def to_one_hot(self, labels):
        vec_size = torch.max(labels);
        one_hot_labels = torch.zeros((labels.size(0), vec_size+1));

        for i in range(labels.size(0)):
            vec = torch.zeros((vec_size+1));
            vec[labels[i]] = 1;
            one_hot_labels[i] = vec;
        
        return one_hot_labels;


    def __len__(self):
        return self.train_len;

    def __getitem__(self, idx):

        input, label = self.train_data[idx], self.train_labels[idx];
        input = torch.reshape(input,(28,28,1)).permute(2,0,1).float();
        label = label.long();
        return input, label;