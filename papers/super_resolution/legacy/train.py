import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

import argparse
import numpy as np

from net import SRNet

TRAINING_SIZE = 0 ;

# parsing command line params
parser = argparse.ArgumentParser() ;
parser.add_argument('--batch_size',type=int, help='batch size') ;
parser.add_argument('--epochs',type=int, help='total number of epochs')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--model_num',type=int, help='epoch to resume',default=-1)
parser.add_argument('--gpu',type=str, help='gpu id')
parser.add_argument('--lr', type=float, help='learning rate')
args = parser.parse_args()

# method to initialize weights
def init_weights(m) :

	if type(m) == nn.Conv2d :
		init.xavier_uniform(m.weight, gain=np.sqrt(2.0)) ;
		init.normal(m.bias) ;

# load training/test data
def load_data() :

	global TRAINING_SIZE ;
	trainx, trainy = np.float32(np.load('../data/train.npy')), np.float32(np.load('../data/train_labels.npy')) ;
	trainx.shape, trainy.shape = (-1,trainx.shape[3], trainx.shape[1], trainx.shape[2]), (-1,trainx.shape[3], trainx.shape[1], trainx.shape[2]) ;
	testx, testy = np.float32(np.load('../data/val.npy')), np.float32(np.load('../data/val_labels.npy')) ;
	testx.shape, testy.shape = (-1, testx.shape[3], testx.shape[1], testx.shape[2]), (-1, testx.shape[3], testx.shape[1], testx.shape[2]);
	
	#print(np.amax(trainy),np.amin(trainy))
	#print(np.array_equal(trainx[0], trainy[0]))

	TRAINING_SIZE = trainx.shape[0] ;
	print('Training size', trainx.shape, trainy.shape) ;
	print('Test size', testx.shape, testy.shape) ;

	return trainx, trainy, testx, testy ;

# shuffle data
def shuffle(x, y) :

	idxs = range(x.shape[0]) ;
	np.random.shuffle(idxs) ;
	xshuffled = x[idxs] ;
	del x;
	yshuffled = y[idxs] ;
	del y;
	return xshuffled, yshuffled ;


log_mode = 'w';
if args.gpu == "cpu":
	device = "cpu";
else:
	device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
# resume training
if args.resume:
	log_mode = 'a';
        model = torch.load('../models/weights' + str(args.model_num) + '.pt',map_location={'cuda:0':'cuda:' + args.gpu});
	print('Resuming training from epoch ' , args.model_num);
else :
	model = SRNet() ;
	model.apply(init_weights) ;
	print('Starting a new training');
model.to(device);

criterion = nn.MSELoss() ;
optimizer = optim.SGD(model.parameters(), lr=args.lr) ;

# hyperparameters
EPOCHS = args.epochs ;
BATCH_SIZE = args.batch_size ;

# data load
print('Loading data') ;
trainx, trainy, testx, testy = load_data() ;
STEPS = TRAINING_SIZE / BATCH_SIZE ;
print('Total number of steps', STEPS);

log = open('../logs/log.txt',log_mode);

for epoch in range(args.model_num + 1, EPOCHS) :

	trainx, trainy = shuffle(trainx, trainy) ;
	cost = 0;
	for step in range(STEPS) :

		idx = step * BATCH_SIZE ;
		binputs, blabels = trainx[idx : idx + BATCH_SIZE], trainy[idx : idx + BATCH_SIZE] ;

		binputs, blabels = Variable(torch.from_numpy(binputs).to(torch.float32)), Variable(torch.from_numpy(blabels).to(torch.float32)) ;	
		optimizer.zero_grad() ;
		
		binputs, blabels = binputs.to(device), blabels.to(device);
		_blabels = model(binputs) ;
		loss = criterion(_blabels, blabels) ;
		loss.backward() ;
		optimizer.step() ;

		cost += loss.data.cpu().numpy();

	print('Epoch ', epoch ,' Loss : ', cost / STEPS) ;
	log.write('Epoch ' +  str(epoch) + ' Loss : ' + str(cost / STEPS) + '\n');
	log.flush();
	torch.save(model, '../models/weights' + str(epoch) + '.pt') ;

log.close();
