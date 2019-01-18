import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy.random as rng
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from sklearn.utils import shuffle

from helpers import *

import numpy as np
import json

RES = 100;
DATA_DIR = '../data'

def load_test_data():

	imgs = np.load('../data/validation_imgs.npy')
	labels = np.load('../data/validation_labels.npy');
	print(imgs.shape, labels.shape);

	embeddings = np.load('../data/validation_embeddings.npy').reshape(imgs.shape[0],-1);
	print('Embedding shape {}'.format(embeddings.shape));

	return imgs, labels, embeddings;


def calculate_accuracy(predict_issame, actual_issame):

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/predict_issame.size
    return tpr, fpr, acc


def infer_by_sigmoid():

	for i in range(val_imgs.shape[0]):
		
		preds[val_labels[i]] = [];
		anchor_img = np.float16([[val_imgs[i][0]]*batch_size]).reshape(batch_size, 100, 100, 3);
		print('Comparing {} with all'.format(val_labels[i]));
		idx = 0;

		while(idx < val_imgs.shape[0]):

			sample_img = val_imgs[idx:idx + batch_size,1];
			batch_len = sample_img.shape[0];
			anchor_img = anchor_img[:batch_len];			
			probs = model.predict([anchor_img, sample_img]).reshape(batch_len);

			for j in range(batch_len):
				if(probs[j] >= 0.5):
					preds[val_labels[i]].append([val_labels[j+batch_len], probs[j]]);
			idx += batch_size;		

		print(preds[val_labels[i]]);
		break;


# method to store embeddings for all TOPS products in the dataset
def store_embeddings(model):

	model = Model(inputs=model.input, outputs=model.layers[2].get_output_at(1));

	imgs = np.load('../data/imgs.npy')
	labels = np.load('../data/labels.npy');
	print(imgs.shape, labels.shape);

	embeddings = [];
	# save embeddings for all crops
	for i in range(imgs.shape[0]):
		embedding = model.predict([imgs[i][0].reshape(1,100,100,3), imgs[i][0].reshape(1,100,100,3)]);
		embeddings.append(embedding);

		if i % 10000 == 0:
			print(i);

	embeddings = np.float16(embeddings);
	print(embeddings.shape);
	np.save('../data/embeddings.npy', embeddings);

# method to compare products('size' rows compared against each other)
def compare_products(embd, size):

	embeddings = embd[:size];
	print(embeddings.shape);
	preds = {};
	for i in range(size):
		if labels[i] not in preds:
			preds[labels[i]] = [];

		distance = np.sum(np.square(embeddings - embeddings[i].reshape(1,-1)), axis=1);
		indices = np.where(distance <= 2.0);

		for j in indices:
			j = j[0];
			probs = model.predict([imgs[i][0].reshape(1,100,100,3), imgs[j][1].reshape(1,100,100,3)])[0][0];
			preds[labels[i]].append([labels[j], str(probs)]);

		print('{} - {} done'.format(i, labels[i]));
		with open('../results/produrlid.json','w') as f:
			json.dump(preds, f);

# method to create json for pids from prod url ids
def create_pid_json():

	with open('../results/produrlid.json') as f:
		preds = json.load(f);
	print(len(preds));

	results = {};
	for key, val in preds.items():
		for pid in prodUrl2pid[key]:
			if pid not in results:
				results[pid] = [];
			for prod in val:
				for pid2 in prodUrl2pid[prod[0]]:
					if pid == pid2:
						continue;
					results[pid].append([pid2, prod[1]]);

	with open('../results/test.json','w') as f:
		json.dump(results, f);


# method to randomly sample pairs and find distance between them -- L2 distance is used
def infer_by_distance(thresh, sample_size):

	# randomly choose 'sample_size' pairs to measure accuracy
	pairs = np.random.choice(embeddings.shape[0],size=(sample_size,2),replace=False);
	cnt = 100;
	labels = np.arange(sample_size);
	for i, pair in enumerate(pairs):
		labels[i] = False;
		if cnt > 0 and np.random.choice([True,False]):
			cnt -= 1;
			pair[1] = pair[0];# setting both class same(for positive class)
		
		# true if same product
		if pair[1] == pair[0]:
			labels[i] = True;	

	x = pairs[:,0]
	y = pairs[:,1]
	preds = np.sum(np.square(embeddings[x] - embeddings[y]), axis=1);

	preds[preds <= thresh] = True;
	preds[preds > thresh] = False;

	print('Threshold {} -- {}'.format(thresh, calculate_accuracy(preds, labels)));


if __name__ == '__main__':

	# loading test data
	imgs, labels, embeddings = load_test_data();
	imgs, labels, embeddings = shuffle(imgs, labels, embeddings);

	# loading model
	model_path = '../weights/2018:10:22:20:20:59/weights-99.hdf5';	
	model = load_model(model_path);

	#infer_by_sigmoid();

	'''
	j = 0.5;
	while j < 10.0:
		infer_by_distance(j, 10000);
		j += 0.5;
	'''

	compare_products(embeddings, 50);
	create_pid_json();