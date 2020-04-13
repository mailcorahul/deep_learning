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

        for i in range(len(self.image_ids)):
            self.image_ids[i] = os.path.splitext(self.image_ids[i])[0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        input_image = Image.open(os.path.join(self.input_dir, image_id + '.jpg')).convert('RGB')
        target_image = Image.open(os.path.join(self.output_dir, image_id + '.png')).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        if self.use_gpu:
            input_image = input_image.cuda()
            target_image = target_image.cuda()

        return input_image, target_image