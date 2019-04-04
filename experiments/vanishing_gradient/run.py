import sys
 
import torch
import torch.nn as nn
from dataloader import MNISTDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from net import Net

EPOCHS = 500;
BATCH_SIZE = 512;

LAYERS_TO_SAVE = [0, 1, 2, 3];
PREV_PARAMS = {};
CHANGE_PARAMS = {};

def get_grad(epoch):

    for name, params in net.named_parameters():        
        if 'weight' not in name and 'bias' not in name:
            continue;
        dparams = 0;
        if epoch == 0:
            PREV_PARAMS[name] = params;
        else:
            dparams = torch.sum(torch.abs(params - PREV_PARAMS[name]));
            CHANGE_PARAMS[name] = dparams;
            PREV_PARAMS[name] = params;
    
        print('Param {}, dparams {}'.format(
            name, dparams));


def train(net, mnist_loader):
    """
    Trains 'net' for epoch 'EPOCHS' using dataloader 'mnist_loader'
    """

    criterion = nn.CrossEntropyLoss();
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    STEPS = len(mnist_dataset)//BATCH_SIZE;
    print('Total number of steps - {}'.format(STEPS));

    for epoch in range(EPOCHS):        
        running_loss = 0.;    
        pbar = tqdm(total=STEPS)
        for i, batch in enumerate(mnist_loader):
            pbar.update(1);

            # get the inputs
            inputs, labels = batch

            inputs = inputs.to(device);
            labels = labels.to(device);
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item();
        pbar.close()
        print('Epoch - {} --> Loss - {}'.format(epoch+1, running_loss/STEPS));
        #get_grad(epoch)
        

if __name__ == '__main__':

    mnist_dataset = MNISTDataset();
    mnist_loader = DataLoader(mnist_dataset, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2);
    net = Net();

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    net.to(device);

    train(net, mnist_loader);
