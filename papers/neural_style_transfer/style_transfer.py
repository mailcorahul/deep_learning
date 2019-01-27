import cv2
import sys
import os
import numpy as np
import scipy.ndimage.filters

from functools import partial

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from helpers import *
from constants import *

# class which stores forward pass outputs of vgg using a forward hook
class SaveFeatures(nn.Module):
	features = None;
	def __init__(self, m):
		self.hook = m.register_forward_hook(self.hook_fn);
	def hook_fn(self, module, input, output):
		self.features = output;
	def close(self):
		self.hook.remove();

# method to read content/style images
def get_images(idx):

	cnt = read_img(os.path.join(IMG_DIR, 'content_{}.jpg'.format(idx)));
	style = read_img(os.path.join(IMG_DIR, 'style_{}.jpg'.format(idx)));

	cnt = np.transpose(cnt, (2,0,1));
	style = np.transpose(style, (2,0,1));

	# show_img(cnt);
	# show_img(style);

	return cnt, style;

# MSE loss between content-input features
def content_loss(yhat):
	_loss = 0;
	for i in range(len(yhat)):
		_loss += F.mse_loss(cnt_features[i], yhat[i]);

	return _loss/len(yhat);

# compute gram matrix
def gram(x):
	b,c,h,w = x.size();
	x = x.view(b*c, -1);
	return torch.mm(x, x.t())

# MSE loss between gram matrices of content and style images
def style_loss(yhat):

	_loss = 0;
	for i in range(len(yhat)):
		style_gram = gram(sty_features[i]);
		ip_gram = gram(yhat[i]);
		_loss += F.mse_loss(style_gram, ip_gram);

	return _loss/len(yhat);

def step():

	c_w = 1;
	s_w = 0.1;
	global i;
	vgg(ip);
	cnt_ip_features = [sf.features.clone() for sf in cnt_sfs];
	sty_ip_features = [sf.features.clone() for sf in sty_sfs];
	cnt_loss = c_w * content_loss(cnt_ip_features);
	sty_loss = s_w * style_loss(sty_ip_features);
	loss = cnt_loss + sty_loss;
	optimizer.zero_grad();
	loss.backward();

	if i % 100 == 0:
		print('Step - {}, Content loss - {}, Style loss - {}, Total Loss - {}'\
			.format(i, cnt_loss.data[0], sty_loss.data[0], loss.data[0]));
		out_img = ip.data.cpu().squeeze().permute(1,2,0).numpy();
		cv2.imwrite(os.path.join('debug', str(i) + '.png'), out_img*255);

	i += 1;
	return loss;


if __name__ == '__main__':


	gpu0 = torch.device("cuda:0")

	MAX = len(os.listdir(IMG_DIR))/2;
	print('Reading content, style images...');
	np_cnt, np_style = get_images(IDX);

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]);

	print('Loading {} model...'.format(MODEL));
	model = getattr(models, MODEL);
	vgg = model(pretrained=True);
	cnt, style = torch.Tensor(np_cnt), torch.Tensor(np_style);		

	# picking model params upto 38 layers leaving out classifier and freezing weights
	vgg = nn.Sequential(*list(vgg.features.children())[:43]).cuda();
	for p in vgg.parameters():
		p.requires_grad = False;

	# register forward hook for content image
	layers = [5, 12, 22];
	cnt_sfs = [SaveFeatures(vgg[i]) for i in layers];

	# pass content/style image and save features
	vgg(Variable(cnt[None].to(gpu0)));
	cnt_features = [sf.features.clone() for sf in cnt_sfs];

	layers = [5, 12, 22, 32, 42];
	sty_sfs = [SaveFeatures(vgg[i]) for i in layers];
	vgg(Variable(style[None].to(gpu0)));
	sty_features = [sf.features.clone() for sf in sty_sfs];

	# input noise image and set it trainable
	np_ip = np.random.uniform(0.0, 1.0, size=np_cnt.shape);
	np_ip = scipy.ndimage.filters.median_filter(np_ip, [8,8,1]);
	ip = torch.Tensor(np_ip)[None].to(gpu0);
	ip = Variable(ip, requires_grad=True);

	optimizer = optim.LBFGS([ip], lr=1);

	print('\nTraining...');
	i = 0;
	while i < STEPS:
		optimizer.step(step);