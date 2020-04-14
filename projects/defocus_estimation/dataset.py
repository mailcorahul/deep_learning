import os
from PIL import Image

from torch.utils.data import Dataset

class BlurDetection(Dataset):

    def __init__(self, root_dir=None, transform=None, use_gpu=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_gpu = use_gpu
        self.input_dir = os.path.join(self.root_dir, 'input')
        self.output_dir = os.path.join(self.root_dir, 'output')
        self.image_ids = os.listdir(self.input_dir)

        # create a list of all image ids present in the dataset folder
        for i in range(len(self.image_ids)):
            self.image_ids[i] = os.path.splitext(self.image_ids[i])[0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # read input and output image
        # input image should have 3 channels and output image should have 1 channel
        image_id = self.image_ids[index]
        input_image = Image.open(os.path.join(self.input_dir, image_id + '.jpg')).convert('RGB')
        target_image = Image.open(os.path.join(self.output_dir, image_id + '.png')).convert('L')

        # apply transform if provided
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        # move tensors to gpu if 'use_gpu' flag is set
        if self.use_gpu:
            input_image = input_image.cuda()
            target_image = target_image.cuda()

        return input_image, target_image