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

def content_loss(yhat):
	return F.mse_loss(cnt_features, yhat);

def gram(x):
	b,c,h,w = x.size();
	x = x.view(b*c, -1);
	return torch.mm(x, x.t());

def style_loss(yhat):

	style_gram = gram(sty_features);
	ip_gram = gram(yhat);

	return F.mse_loss(style_gram, ip_gram);

def step():

	global i;
	vgg(ip);
	ip_features = sfs.features;
	cnt_loss = content_loss(ip_features);
	sty_loss = style_loss(ip_features);
	loss = sty_loss;
	optimizer.zero_grad();
	loss.backward();

	if i % 100 == 0:
		print('Step - {}, Loss - {}'.format(i, loss.data[0]));
		out_img = ip.data.cpu().squeeze().permute(1,2,0).numpy();
		cv2.imwrite(os.path.join('debug', str(i) + '.png'), out_img*255);

	i += 1;
	return loss;


if __name__ == '__main__':


	gpu0 = torch.device("cuda:0")

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
	vgg = nn.Sequential(*list(vgg.features.children())[:41]).cuda();
	for p in vgg.parameters():
		p.requires_grad = False;

	# register forward hook for layer 40
	sfs = SaveFeatures(vgg[26]);

	# pass content/style image and save features
	vgg(Variable(cnt[None].to(gpu0)));
	cnt_features = sfs.features;

	vgg(Variable(style[None].to(gpu0)));
	sty_features = sfs.features;

	print(cnt_features.size());

	# input noise image and set it trainable
	np_ip = np.random.uniform(0.0, 1.0, size=np_cnt.shape);
	np_ip = scipy.ndimage.filters.median_filter(np_ip, [8,8,1]);
	ip = torch.Tensor(np_ip)[None].to(gpu0);
	ip = Variable(ip, requires_grad=True);

	optimizer = optim.LBFGS([ip], lr=1);

	print('Training...');
	i = 0;
	while i < STEPS:
		optimizer.step(step);