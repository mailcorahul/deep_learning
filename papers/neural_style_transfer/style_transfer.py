import cv2
import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from helpers import *
from constants import *

def get_images(idx):

	cnt = read_img(os.path.join(IMG_DIR, 'content_{}.jpg'.format(idx)));
	style = read_img(os.path.join(IMG_DIR, 'style_{}.jpg'.format(idx)));

	# show_img(cnt);
	# show_img(style);

	return cnt, style;

if __name__ == '__main__':

	MAX = len(os.listdir(IMG_DIR))/2;
	print('Reading content, style images...');
	cnt, style = get_images(np.random.randint(MAX));

	print('Loading {} model...'.format(MODEL));
	model = getattr(models, MODEL);
	vgg = model(pretrained=True);

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]);

