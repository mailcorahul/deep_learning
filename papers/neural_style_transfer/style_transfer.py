import cv2
import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from helpers import *
from constants import *

class SaveFeatures(nn.Module):
	features = None;
	def __init__(self, m):
		self.hook = m.register_forward_hook(self.hook_fn);
	def hook_fn(self, module, input, output):
		self.features = output;
	def close(self):
		self.hook.remove();

def get_images(idx):

	cnt = read_img(os.path.join(IMG_DIR, 'content_{}.jpg'.format(idx)));
	style = read_img(os.path.join(IMG_DIR, 'style_{}.jpg'.format(idx)));

	cnt = np.transpose(cnt, (2,0,1));
	style = np.transpose(style, (2,0,1));

	# show_img(cnt);
	# show_img(style);

	return cnt, style;

if __name__ == '__main__':

	MAX = len(os.listdir(IMG_DIR))/2;
	print('Reading content, style images...');
	np_cnt, np_style = get_images(np.random.randint(MAX));

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]);

	print('Loading {} model...'.format(MODEL));
	model = getattr(models, MODEL);
	vgg = model(pretrained=True);
	cnt, style = torch.Tensor(np_cnt), torch.Tensor(np_style);	

	# picking model params upto 38 layers leaving out classifier and freezing weights
	vgg = nn.Sequential(*list(vgg.features.children())[:41]);
	for p in vgg.parameters():
		p.requires_grad = False;

	# register forward hook for layer 40
	sfs = SaveFeatures(vgg[40]);

	# pass content image and save features
	vgg(Variable(cnt[None]));
	cnt_features = sfs.features;

	# input noise image and set it trainable
	np_ip = np.random.uniform(0.0, 1.0, size=np_cnt.shape);
	ip = torch.Tensor(np_ip)[None];
	ip = Variable(ip, requires_grad=True);

	criterion = nn.MSELoss();
	optimizer = optim.SGD([ip], lr=0.01);

	for i in range(STEPS):
		vgg(ip);
		ip_features = sfs.features;
		loss = criterion(ip_features, cnt_features);
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		if i % 10 == 0:
			print(loss.data);
			out_img = ip.data.squeeze().permute(1,2,0).numpy();
			print(np.amax(out_img), np.amin(out_img));
			cv2.imwrite(os.path.join('debug', str(i) + '.png'), out_img*255);