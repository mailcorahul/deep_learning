from helpers import *
import pandas as pd
import requests
import os
import numpy as np
import json
import cv2
import ast
import random

DATA_DIR = '../data'; 
RES = 100;
TRAIN_VAL_SPLIT = 0.8;

def split_data():

	files = os.listdir(os.path.join(DATA_DIR, 'TOPS'));
	random.shuffle(files);
	train_files = files[:int(len(files)*TRAIN_VAL_SPLIT)];
	val_files = files[int(len(files)*TRAIN_VAL_SPLIT):]

	print(len(train_files), len(val_files));
	print(len(set(train_files + val_files)));

	with open(os.path.join(DATA_DIR, 'train_pids.txt'), 'w') as f:
		f.write(str(train_files));

	with open(os.path.join(DATA_DIR, 'validation_pids.txt'), 'w') as f:
		f.write(str(val_files));


# creating two examples per product --> one - original image, two - randomly rotated
def create_examples(_type):

	with open(os.path.join(DATA_DIR, _type + '_pids.txt')) as f:
		files = ast.literal_eval(f.read());

	print('Total number of images {}'.format(len(files)));
	images = np.zeros((len(files), 2, RES, RES, 3), dtype=np.float16);
	print('Generating ', images.shape);
	prods = [];

	for i, file in enumerate(files):
		try:
			pid = os.path.splitext(file)[0];
			img = read_image(os.path.join(DATA_DIR, 'TOPS', file));
			img = cv2.resize(img, (RES, RES));
			images[i,0] = np.float16(img)/255; # original image
			images[i,1] = np.float16(random_rotation(img, 15))/255; # randomly rotated
			prods.append(pid);

			if i % 5000 == 0:
				print(i, pid);

		except Exception as e:
			print(file);
		
	print(images.shape);
	np.save(os.path.join(DATA_DIR, _type + '_imgs.npy'), images);
	np.save(os.path.join(DATA_DIR, _type + '_labels.npy'), prods);

# method to save images of resolution resl x resl for any category
def download_images(json_name):
	
	with open(os.path.join(DATA_DIR, json_name)) as f:
		data = json.load(f);
	
	print('Total number of rows - {}'.format(len(data)));

	with open(os.path.join(DATA_DIR, 'redownload.txt')) as f:
		redownload = ast.literal_eval(f.read());

	prods = [];
	images = [];
	for i, (prodUrl, urls) in enumerate(data.items()):
		try:

			if prodUrl not in redownload:
				continue;

			if i % 1000 == 0:
				print('Downloading product image {} , count {}'.format(prodUrl, i + 1));

			imageurl = urls['-1']; # specified resolution
			bytes = download_from_url(imageurl);						
			imgpath = os.path.join(DATA_DIR, 'TOPS', prodUrl + '.jpeg')
			write_image(bytes, imgpath);			
			img = read_image(imgpath);
			print(img.shape);
			# img = cv2.resize(img, (RES, RES));
			prods.append(prodUrl);
			images.append(img);

		except Exception as e:
			print(e, prodUrl, imageurl);

	'''
	np.save(os.path.join(DATA_DIR, 'tops_imgs.npy'), images);
	np.save(os.path.join(DATA_DIR, 'tops_prods.npy'), prods);
	'''

def extract_unique_products(category):

	data = pd.read_csv(os.path.join(DATA_DIR, category + '.csv'));
	print('Total number of rows - {}'.format(len(data)));
	print(data.head());
	prodUrl2pId = {};
	prodUrl2imgUrl = {};

	for i in range(len(data)):
		try:
			row = data.iloc[i];
			prodUrl = row['productUrl'];
			prodUrl = prodUrl[:prodUrl.index('?pid=')].split('/')[-1];
			imageurl = row['imageUrlStr'].split(';');

			if prodUrl not in prodUrl2pId:
				prodUrl2pId[prodUrl] = [];
				prodUrl2imgUrl[prodUrl] = {};

			prodUrl2pId[prodUrl].append(row['productId']);			
			prodUrl2imgUrl[prodUrl][-1] = imageurl[0];
			prodUrl2imgUrl[prodUrl][200] = imageurl[1];
			prodUrl2imgUrl[prodUrl][400] = imageurl[2];
			prodUrl2imgUrl[prodUrl][800] = imageurl[3];

		except Exception as e:
			print(e, i);

	print('Total number of unique products in {} - {}'.format(category, len(prodUrl2pId)));

	with open(os.path.join(DATA_DIR, category + '_produrl2pid.json'), 'w') as f:
		json.dump(prodUrl2pId, f);

	with open(os.path.join(DATA_DIR, category + '_produrl2imgurls.json'), 'w') as f:
		json.dump(prodUrl2imgUrl, f);
	

# method to extract a particular category from column 'categories'
def extract_category(data, category_name, category_str):
	extData = [];
	for i in range(len(data)):
		try:
			row = data.iloc[i]
			category = row['categories']
			_type = category.split('>')[-1];
			if category_str in _type.lower():
				extData.append(row);
		except Exception as e:
			print(category, i)

	extDataDf = pd.DataFrame(extData);
	print('Total number of rows in {} - {}'.format(category_name, len(extDataDf)));
	extDataDf.to_csv(os.path.join(DATA_DIR, category_name + '.csv'));

def extract_pid2produrlid_mapping():

	pid2produrl = []
	with open(os.path.join(DATA_DIR, 'TOPS_produrl2pid.json')) as f:
		produrl2pid = json.load(f);

	print('Number of unique products -- {}'.format(len(produrl2pid)));
	for produrl, pids in produrl2pid.items():
		for pid in pids:
			pid2produrl.append([pid, produrl]);
	
	df = pd.DataFrame(pid2produrl, columns=['pid','produrlid']);
	print('Number of pids -- {}'.format(len(df)));
	df.to_csv(os.path.join(DATA_DIR, 'TOPS_pid2produrl.csv'));

if __name__ == '__main__':

	# data = pd.read_csv(os.path.join(DATA_DIR, "2oq-c1r.csv"));
	# print('Total number of rows OVERALL - {}'.format(len(data)));

	# extract_category(data, 'TOPS', 'top'); # extracting a particular category -- in this case "top"
	# extract_category(data, 'TUNICS', 'tunic'); # extracting a particular category -- in this case "top"

	# extract_unique_products('TOPS');	
	# download_images('TOPS_produrl2imgurls.json');
	# check_images();

	#split_data();	
	# create_examples('validation');
	# create_examples('train');

	extract_pid2produrlid_mapping();
