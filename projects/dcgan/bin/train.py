import os
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
from nets.GAN import Discriminator, Generator


gpu0 = torch.device("cuda:0")
os.makedirs('/tmp/dcgan', exist_ok=True)

def visualize(net, num_images=10):
    """
    Visualizes outputs for random set of test images
    """

    inputs = generate_random_noise((1, 9, 9), num_images)

    # forward pass
    outputs = net(inputs.to(gpu0))

    # iterate over and save outputs to disk
    for i, output in enumerate(outputs):
        output = output.cpu().data.numpy().reshape(28, 28)*255.
        cv2.imwrite('/tmp/dcgan/{}.png'.format(i), output)

def generate_random_noise(input_size, batch_size):

    ch, h, w = input_size
    rnd_batch = torch.zeros(batch_size, ch, h, w).normal_(0, 1)

    return rnd_batch

def set_grad(net, trainable):

    for p in net.parameters():
        p.requires_grad = trainable


def train(generator_net, discriminator_net, dataset, batch_size=256, num_epochs=1000):
    """
    Trains a neural net

    Args:
        net: Neural Network class
        dataset: Dataset class defining train/test dataset
        batch_size: Number of input-output pairs per batch(step)
        num_epochs: Number of epochs to train
    """

    # define dataloader
    _dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader = iter(_dataloader)

    # define criterion
    bce_loss = nn.BCEWithLogitsLoss()
    
    # move to gpu
    generator_net = generator_net.cuda()
    discriminator_net = discriminator_net.cuda()

    # define optim
    gen_optim = optim.Adam(generator_net.parameters())
    disc_optim = optim.Adam(discriminator_net.parameters())

    steps = len(dataset)//batch_size
    #steps = 10
    print('Total number of steps - {}'.format(steps))

    gen_real_labels = torch.ones(batch_size, 1)
    gen_fake_labels = torch.zeros(batch_size, 1)

    for epoch in range(num_epochs):    
        disc_loss = 0.
        gen_loss = 0.  
        pbar = tqdm(total=steps*2)

        for step_idx in range(steps*2):
            pbar.update(1)
            # train discriminator
            if True or step_idx == 0 or step_idx%25 != 0:
                #print('[/] training discriminator...')

                # freeze generator
                set_grad(generator_net, trainable=False)
                set_grad(discriminator_net, trainable=True)

                try:
                    real_inputs, real_labels = next(dataloader)
                except StopIteration:
                    dataloader = iter(_dataloader)
                    real_inputs, real_labels = next(dataloader)

                rnd_batch = generate_random_noise((1, 9, 9), batch_size)

                # move to gpu
                rnd_batch = rnd_batch.cuda()

                fake_inputs = generator_net(rnd_batch)
                fake_labels = gen_fake_labels

                real_inputs, real_labels = real_inputs.cuda(), real_labels.cuda()
                fake_inputs, fake_labels = fake_inputs.cuda(), fake_labels.cuda()

                # zero the parameter gradients
                gen_optim.zero_grad()
                disc_optim.zero_grad()

                # forward + backward + optimize
                real_outputs = discriminator_net(real_inputs)
                fake_outputs = discriminator_net(fake_inputs)
                real_loss = bce_loss(real_outputs, real_labels)
                fake_loss = bce_loss(fake_outputs, fake_labels)
                loss = real_loss + fake_loss
                loss.backward()
                disc_optim.step()

                disc_loss += loss.item()

            # train generator
            else:
                print('[/] training generator...')
                # freeze discriminator
                set_grad(discriminator_net, trainable=False)
                set_grad(generator_net, trainable=True)

                rnd_batch = generate_random_noise((1, 9, 9), batch_size)

                # move to gpu
                rnd_batch = rnd_batch.cuda()

                fake_inputs = generator_net(rnd_batch)
                fake_inputs = fake_inputs.cuda()
                fake_labels = gen_real_labels
                fake_labels = fake_labels.cuda()
                
                disc_optim.zero_grad()
                gen_optim.zero_grad()
                fake_outputs = discriminator_net(fake_inputs)
                loss = bce_loss(fake_outputs, fake_labels)
                loss.backward()
                gen_optim.step()

                gen_loss += loss.item()


        pbar.close()
        print('Epoch - {} --> Generator Loss - {:.4f}, Discriminator loss - {:.4f}'.format(
            epoch+1, gen_loss*2/steps, disc_loss*2/steps))

        # visualize generator output after every epoch
        visualize(generator_net)

if __name__ == '__main__':
    
    BATCH_SIZE = 512
    NUM_EPOCHS = 500

    # load dataset
    dataset = MNISTDataset()

    # define generator and discriminator networks
    discriminator = Discriminator(nc=1, nw=28, nh=28, nclasses=1)
    generator = Generator(nc=1, nin_w=9, nin_h=9, nout_w=28, nout_h=28)

    print('\nNetwork summary...')
    print(discriminator)
    print(generator)
    
    # train DCGAN
    train(
        generator,
        discriminator,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS
    )
