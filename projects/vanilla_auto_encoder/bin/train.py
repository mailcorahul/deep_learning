import sys
sys.path.append('.')

from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from data.dataloader import MNISTDataset
from nets.AutoEncoder import AutoEncoder

def visualize(net, dataset, num_images=10):
    """
    Visualizes outputs for random set of test images
    """

    idxs = np.random.randint(dataset.test_data.size(0), size=(num_images,))
    inputs = dataset.test_data[idxs]

    # forward pass
    outputs = net(inputs)

    # iterate over and save outputs to disk
    for i, output in enumerate(outputs):
        output = output.data.numpy().reshape(28, 28)*255.
        cv2.imwrite('/tmp/{}.png'.format(i), output)


def train(net, dataset, batch_size=256, num_epochs=1000):
    """
    Trains a neural net

    Args:
        net: Neural Network class
        dataset: Dataset class defining train/test dataset
        batch_size: Number of input-output pairs per batch(step)
        num_epochs: Number of epochs to train
    """

    # define dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2)

    # define criterion and optim 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    steps = len(dataset)//batch_size;
    print('Total number of steps - {}'.format(steps));

    for epoch in range(num_epochs):        
        running_loss = 0.;    
        pbar = tqdm(total=steps)
        for step_idx, batch in enumerate(dataloader):
            pbar.update(1);

            # get the inputs
            inputs, labels = batch
            #inputs = inputs.to(device);
            #labels = labels.to(device);

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item();            

        pbar.close()
        print('Epoch - {} --> Loss - {:.4f}'.format(epoch+1, running_loss/steps));        
        visualize(net, dataset)


if __name__ == '__main__':
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 1

    # load dataset
    dataset = MNISTDataset()

    # define autoencoder network
    autoencoder = AutoEncoder()

    print('\nNetwork summary...')
    print(autoencoder)

    # train autoencoder
    train(
        net=autoencoder, 
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS
    )
