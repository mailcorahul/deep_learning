import argparse
import torch
from net import DeFocusNet

parser = argparse.ArgumentParser()
parser.add_argument('--train-path', type=str, help='path to train dataset')
parser.add_argument('--test-path', type=str, help='path to test dataset')
parser.add_argument('--batch-size', type=str, help='batch size during training')

args = parser.parse_args()

if __name__ == '__main__':

    defocus_net = DeFocusNet()
    rnd_input = torch.zeros((1, 3, 512, 512)).random_()

    defocus_net(rnd_input)