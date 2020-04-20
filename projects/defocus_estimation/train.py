import os
import argparse
from tqdm import tqdm

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import BlurDetection
from net import DeFocusNet

parser = argparse.ArgumentParser()
parser.add_argument('--image-side', type=int, help="width and height of input image", default=512)
parser.add_argument('--train-path', type=str, help='path to train dataset')
parser.add_argument('--test-path', type=str, help='path to test dataset')
parser.add_argument('--num-epochs', type=int, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, help='batch size during training')
parser.add_argument('--debug-path', type=str, help='path to save test defocus maps for debugging')
parser.add_argument('--use-gpu', action='store_true', help="flag to use gpu")

args = parser.parse_args()

def init():
    """Function to create necessary directories."""

    if args.debug_path is not None:
        os.makedirs(args.debug_path, exist_ok=True)

def train():
    """Function to train Defocus prediction net."""

    image_transform =  transforms.Compose([
            transforms.Resize((args.image_side, args.image_side)),
            transforms.ToTensor()
        ])
    train_dataset = BlurDetection(root_dir=args.train_path, transform=image_transform, use_gpu=args.use_gpu)
    test_dataset = BlurDetection(root_dir=args.test_path, transform=image_transform, use_gpu=args.use_gpu)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    defocus_net = DeFocusNet()
    if args.use_gpu:
        defocus_net.cuda()

    optimizer = optim.Adam(params=defocus_net.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    steps = len(train_dataset)//args.batch_size
    for epoch in range(args.num_epochs):
        epoch_loss = 0.
        pbar = tqdm(total=steps)

        # batch loop for one complete epoch
        for batch_idx, batch in enumerate(train_loader):
            pbar.update(1)
            input_image, target_image = batch
            pred_image = defocus_net(input_image)

            optimizer.zero_grad()
            train_loss = criterion(pred_image, target_image)
            epoch_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
            #break

        pbar.close()
        print('[/] epoch: {} --> train loss: {:.4f}'.format(epoch+1, epoch_loss/steps))

        # create defocus maps on test set
        print('[/] inference on test images...')
        for (batch_inputs, batch_targets) in test_loader:
            output_images = torch.sigmoid(defocus_net(batch_inputs))
            output_images = output_images.squeeze(1).detach().cpu().numpy()

            # save images to disk
            for idx in range(output_images.shape[0]):
                # threshold predicted image
                output_image = output_images[idx]
                pmin, pmax = np.amin(output_image), np.amax(output_image)
                output_image[output_image == pmin] = 0.
                output_image[output_image == pmax] = 255.

                # concat target and predicted image for visualization.
                target_image = batch_targets[idx].squeeze(0).detach().cpu().numpy() * 255.
                vis_image = np.hstack([target_image, output_image])
                cv2.imwrite(os.path.join(args.debug_path, 'test_{}.png'.format(idx)), vis_image)
        print('done.\n')

if __name__ == '__main__':

    print('[/] initializing...')
    init()
    print('[/] training...')
    train()