from constants import *
import cv2

def read_img(path):
	img = cv2.imread(path);
	# resize if image is large
	if img.shape[0] > MAX_H and img.shape[1] > MAX_W:
		img = cv2.resize(img, (MAX_W, MAX_H));

	return img/255.0;

def show_img(img):
	cv2.imshow('img', img);
	cv2.waitKey(0);