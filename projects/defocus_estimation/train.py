import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import BlurDetection
from net import DeFocusNet

parser = argparse.ArgumentParser()
parser.add_argument('--image-side', type=int, help="width and height of input image")
parser.add_argument('--train-path', type=str, help='path to train dataset')
parser.add_argument('--test-path', type=str, help='path to test dataset')
parser.add_argument('--num-epochs', type=int, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, help='batch size during training')
parser.add_argument('--use-gpu', action='store_true', help="flag to use gpu")

args = parser.parse_args()

def train():

    train_transform =  transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    dataset = BlurDetection(root_dir=args.train_path, transform=train_transform, use_gpu=args.use_gpu)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    defocus_net = DeFocusNet()
    if args.use_gpu:
        defocus_net.cuda()

    optimizer = optim.Adam(params=defocus_net.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    steps = len(dataset)//args.batch_size
    for epoch in range(args.num_epochs):
        epoch_loss = 0.
        pbar = tqdm(total=steps)

        for batch_idx, batch in enumerate(dataloader):
            pbar.update(1)
            input_image, target_image = batch
            pred_image = defocus_net(input_image)

            optimizer.zero_grad()
            train_loss = criterion(pred_image, target_image)
            epoch_loss += train_loss
            train_loss.backward()
            optimizer.step()

        pbar.close()
        print('Epoch - {} --> Loss - {:.4f}'.format(epoch+1, epoch_loss/steps))


if __name__ == '__main__':
    train()