#!/usr/bin/env python
# coding: utf-8

from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import argparse
import datetime

RES = 100;
PATH = "/home/zlabs-nlp/backup/raghul/duplicate_detection/data"
weights_path = '/home/zlabs-nlp/backup/raghul/duplicate_detection/weights';    
CHANNEL_AXIS = 3;
dataset = {};

def parse_args():

    parser = argparse.ArgumentParser() ;
    parser.add_argument('--batch-size',type=int, help='batch size');
    parser.add_argument('--epochs',type=int, help='total number of epochs');
    parser.add_argument('--resume', type=bool, help='resume training from a particular epoch', default=False);
    parser.add_argument('--lr',type=float, help='learning rate');
    parser.add_argument('--ts',type=str, help='timestamp to resume');
    parser.add_argument('--epoch-num', type=int, help='resume training from epoch-num', default=0);

    args = parser.parse_args();

    return args;


def load_data():    
    print('Loading train and validation data');
    dataset['train'] = np.load(os.path.join(PATH, "train_imgs.npy"))
    dataset['validation'] = np.load(os.path.join(PATH, "validation_imgs.npy"))
    print(dataset['train'].shape, dataset['validation'].shape);

def create_model(lr):

    input_shape = (RES, RES, 3)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    #build convnet to use in each siamese 'leg'
    convnet = Sequential()
    convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                       kernel_initializer="he_normal",kernel_regularizer=l2(2e-4)))
    convnet.add(BatchNormalization(axis=CHANNEL_AXIS));
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(7,7),activation='relu',
                       kernel_regularizer=l2(2e-4),kernel_initializer="he_normal"))
    convnet.add(BatchNormalization(axis=CHANNEL_AXIS));
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer="he_normal",kernel_regularizer=l2(2e-4)))
    convnet.add(BatchNormalization(axis=CHANNEL_AXIS));
    convnet.add(Dropout(0.2));
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer="he_normal",kernel_regularizer=l2(2e-4)))
    convnet.add(BatchNormalization(axis=CHANNEL_AXIS));
    convnet.add(Dropout(0.2));
    convnet.add(Flatten())
    convnet.add(Dense(512,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer="he_normal"))

    #call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    #layer to merge two encoded inputs with the l1/l2 distance between them
    L1_layer = Lambda(lambda tensors:K.square(tensors[0] - tensors[1]))
    #call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = SGD(lr)
    convnet.summary();
    siamese_net.summary()

    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    siamese_net.count_params()

    return siamese_net;

def concat_images(X):
    """Concatenates a bunch of images into a big matrix for plotting purposes."""
    nc,h,w,_ = X.shape
    X = X.reshape(nc,h,w,3)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n*w,n*h,3))
    x = 0
    y = 0
    for example in range(nc):
        img[x*w:(x+1)*w,y*h:(y+1)*h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img


def plot_oneshot_task(pairs):
    """Takes a one-shot task given to a siamese net and  """
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.matshow(pairs[0][0].reshape(RES,RES, 3),cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# #example of a one-shot learning task
# pairs, targets = loader.make_oneshot_task(20,"train","Japanese_(katakana)")
# plot_oneshot_task(pairs)

class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, path, data_subsets = ["train", "validation"]):
        print("Creating Siamese Network Object and initializing object variables");
        self.data = {}
        self.categories = {}
        self.info = {}
        self.batch_idx = 0;
        self.classes = {};
        
        for name in data_subsets:            
            self.data[name] = dataset[name]
            self.classes[name] = np.arange(dataset[name].shape[0])

    def get_batch(self,batch_size,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X = self.data[s]
        n_classes, n_examples, w, h, d = X.shape

        #randomly sample several classes to use in the batch
        #categories = rng.choice(n_classes,size=(batch_size,),replace=False)
        categories = self.classes[s][self.batch_idx : self.batch_idx + batch_size];
        batch_len = len(categories);
        positives = rng.choice(range(batch_len), size=batch_len//2, replace=False);# randomly choose half indices for positive samples
        pairs = [np.zeros((batch_len, h, w, 3)) for i in range(2)]

        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_len,))
        targets[positives] = 1;
        for i in range(batch_len):
            category = categories[i];
            idx_1 = 0 #rng.randint(0, n_examples)
            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 3)
            idx_2 = rng.randint(0, n_examples)
            #pick images of same class if i in positives arr
            if i in positives:
                category_2 = category  
                idx_2 = 1; # always choosing the rotated example as 2nd pair for positive case
            else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1,n_classes)) % n_classes
            pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h, 3)
                
        return pairs, targets
    
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)    

    def make_oneshot_task(self,N,s="validation",language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X = self.data[s]        
        n_classes, n_examples, w, h, d = X.shape
        indices = rng.randint(0,n_examples,size=(N,))

        if language is not None:
            low, high = 0, X.shape[0];#self.categories[s][language]
            if N > high - low:
                raise ValueError("This language ({}) has less than {} letters".format(language, N))
            categories = rng.choice(range(low,high),size=(N,),replace=False)
            
        else:#if no language specified just pick a bunch of random letters
            categories = rng.choice(range(n_classes),size=(N,),replace=False)            
            
        true_category = categories[0]
        ex1, ex2 = 0,1#rng.choice(n_examples,replace=False,size=(2,))
        test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h, 3)
        support_set = X[categories,indices,:,:]
        support_set[0,:,:] = X[true_category,ex2]        
        support_set = support_set.reshape(N, w, h, 3)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image,support_set]

        return pairs, targets
    
    def test_oneshot(self,model,N,k,s="validation",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        #if verbose:
        #   print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
    
    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size))
    
if __name__ == '__main__':

    # parse args
    args = parse_args();
     
    # loading train and validation data
    load_data();

    # instantiating keras model
    siamese_net = create_model(args.lr);
    start = 0;
    if args.resume:
        print('Resuming training for {} from epoch {}'.format(args.ts, args.epoch_num));
        run_name_dir = os.path.join(weights_path, args.ts);
        siamese_net.load_weights(os.path.join(run_name_dir, 'weights-{}.hdf5'.format(args.epoch_num)));
        start = args.epoch_num + 1;
    else:
        print('Starting a new training');
        run_name_dir = os.path.join(weights_path, datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')) ;
        if not os.path.exists(run_name_dir) :
            os.makedirs(run_name_dir) ;
    
    # Instantiate the class
    loader = Siamese_Loader(PATH)

    print(loader.data['train'].shape,loader.data['validation'].shape,len(loader.classes['train']),len(loader.classes['validation']))

    DATASET_SIZE = dataset['train'].shape[0];
    evaluate_every = 125 # interval for evaluating on one-shot tasks
    loss_every = 500 # interval for printing loss (iterations)
    batch_size = args.batch_size;#32
    EPOCHS = args.epochs;
    N_way = 20 # how many classes for testing one-shot tasks>
    n_val = 250 #how mahy one-shot tasks to validate on?
    best = -1
    STEPS = DATASET_SIZE//batch_size;
    print('Total number of steps {}'.format(STEPS));

    
    for epoch in range(start, EPOCHS):    
        i = 0;
        loss = 0;
        while True:        
            (inputs,targets)=loader.get_batch(batch_size)
            loss += siamese_net.train_on_batch(inputs,targets)
            loader.batch_idx = loader.batch_idx + batch_size;        
            if i != 0 and i % evaluate_every == 0:
                val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
                #best=val_acc                        
                print("iteration {}, training loss: {:.5f},".format(i,loss/i))
            if loader.batch_idx >= DATASET_SIZE:
                loader.batch_idx = 0;
                print('Shuffling dataset');
                loader.data['train'] = shuffle(loader.data['train']);
                break;
            i += 1;
        print('Epoch {} --- Loss {:.5f}'.format(epoch, loss/STEPS));
        siamese_net.save(os.path.join(run_name_dir, 'weights-{}.hdf5'.format(epoch)));
