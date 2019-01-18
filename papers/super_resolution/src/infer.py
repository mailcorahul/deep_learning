from net import SRNet
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import cv2
import os
import numpy as np
import scipy.misc
import shutil
import time
from math import log10, ceil

parser = argparse.ArgumentParser() ;
parser.add_argument('--model', type=str, help='model epoch number to load')
parser.add_argument('--image', type=bool, help='infer raw image',default=False)
parser.add_argument('--npy', type=bool, help='infer npy', default=False)
parser.add_argument('--gpu',type=str, help='gpu id')
args = parser.parse_args() ;


if __name__ == '__main__':

	# loading model
	
	if args.gpu == "cpu":
		device = "cpu";
	else:
		device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
	

	model = torch.load('../models/weights' + args.model + '.pt',map_location={'cuda:2':'cuda:' + args.gpu});
	model = model.eval();
	model.to(device);
	criterion = nn.MSELoss();

	if args.image:
   		file = os.listdir('test');	
		y = np.float32(cv2.imread('test/' + file[0])) / 255;
		y.shape = (1,3,32,100) 
		y = Variable(torch.from_numpy(y))
		_y = model(y)
		_y = _y.data.numpy() * 255;
		_y = np.uint8(_y)
		_y.shape = (32,100,3)
		print(_y.shape)
		cv2.imwrite('test_results/_y' + file[0], _y);
	elif args.npy:
		y = np.float32(np.load('../data/val.npy')).reshape(-1,3,32,100);
		gty = np.float32(np.load('../data/val_labels.npy')).reshape(-1,3,32,100);
		print(y.shape, gty.shape);
		y = Variable(torch.from_numpy(y))
		preds = [];
		print('forward pass');
		
		batch_size = 256;	
		j = 0;
		avg_psnr = 0;
		_len = ceil(float(len(y))/batch_size);
		print('Number of batches', _len);

		while 1:
			by = y[j:j+batch_size].to(device);
			gby = torch.from_numpy(gty[j:j+batch_size]).to(device);
			print(gby.dtype);
			j = (j+batch_size)%y.shape[0];
			start_time = time.time();
			_y = model(by)
			mse = criterion(_y, gby)
			psnr = 10 * log10(1 / mse.item())
			avg_psnr += psnr
			print('Time taken for a batch', time.time() - start_time);
			_y = _y.cpu().data.numpy() * 255;
			_y = np.uint8(_y)
			preds += list(_y);
			if j < batch_size:
			 break; 
	
		print('Average PSNR:', avg_psnr/_len);
	
		_y = np.uint8(preds);
		print('_y shape', _y.shape);
		if os.path.exists('../results'):
			shutil.rmtree('../results');
		os.makedirs('../results/_y');
		os.makedirs('../results/y');
		for i, img in enumerate(_y):
			scipy.misc.imsave('../results/_y/' + str(i) + '.png', img.reshape(32,100,3));
		for i, img in enumerate(gty):
			scipy.misc.imsave('../results/y/' + str(i) + '.png', img.reshape(32,100,3));
 


