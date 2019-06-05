import sys
sys.path.append('.')

import numpy as np
import cv2
from data.dataloader import MNISTDataset
from torch.utils.data import DataLoader
import torch

class DataValidator():
    """Class to validate DataLoader"""
    def __init__(self):
        BATCH_SIZE = 1
        self.dataset = MNISTDataset()
        self.loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=1)

    def visualize(self, num_images=50):
        """
        Visualize dataset
        """

        for i, batch in enumerate(self.loader):
            input, label = batch
            input = input.numpy().reshape(28, 28)#*255
            label = label.numpy().reshape(28, 28)#*255
            
            # imshow input-output pairs            
            cv2.imshow('input', input)
            cv2.imshow('label', label)
            cv2.waitKey(0)

            if i == num_images - 1:
                break

    def statistics(self):
        """
        Get statistics for a given dataset
        """

        mean_image = torch.mean(self.dataset.train_data, dim=0)
        print('\nMean image statistics...')
        print('Size: {}'.format(mean_image.size()))

        mean_image = mean_image.numpy().reshape(28, 28)#*255
        print('Min value: {}, Max value: {}'.format(
            np.min(mean_image), np.max(mean_image)))
        cv2.imshow('mean_image', mean_image)
        cv2.waitKey(0)

        print('\nEntire dataset statistics...')
        print('Min pixel value: {}, Max pixel value: {}'.format(
            torch.min(self.dataset.train_data), 
            torch.max(self.dataset.train_data)))
