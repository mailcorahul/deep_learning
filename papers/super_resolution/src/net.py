import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.misc
import cv2

class SRNet(nn.Module) :

	def __init__(self):
		super(SRNet, self).__init__() ;
		self.conv1 = nn.Conv2d(3, 64, 9, padding=4) ; 
		self.conv2 = nn.Conv2d(64, 32, 1) ; 
		self.conv3 = nn.Conv2d(32, 3, 5, padding=2) ; 

	def forward(self, x) :
		
		x = F.relu(self.conv1(x)) ;
		x = F.relu(self.conv2(x)) ;
		x = F.sigmoid(self.conv3(x)) ;

		return x ;
