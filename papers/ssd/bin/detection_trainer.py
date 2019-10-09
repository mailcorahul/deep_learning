import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

class DetectionTrainer:
    """An Object Detection Trainer class for PyTorch."""

    def __init__(self, 
        train_dataset=None, 
        test_dataset=None,
        net=None,
        optimizer=None, 
        class_criterion=None, 
        box_criterion=None):
        """Initialize a DetectionTrainer object.

        Args:
            train_dataset(torch.utils.data.Dataset):  Train Dataset.
            test_dataset(torch.utils.data.Dataset):  Test Dataset.
            net(torch.nn.Module):   Object Detection Network architecture.
            optimizer(torch.optim): Training optimizer.
            class_criterion(torch.nn): Criterion for object classification.
            box_criterion(torch.nn): Criterion for bbox regression.

        Returns:
            None
        """

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.net = net
        self.optimizer = optimizer
        self.class_criterion = class_criterion
        self.box_criterion = box_criterion        
        self.epoch_idx = 0
        self.loss = {'train': -1., 'train_cls': -1., 'train_reg': -1., 'test': -1., 'test_cls': -1., 'test_reg': -1.}
        self.metrics = []


    def adjust_lr(self):
        pass

    def train(self, 
        num_epochs=100, 
        batch_size=128, 
        shuffle=True,
        num_workers=4,
        callbacks=[]):
        """Defines the training loop for a PyTorch model.
        
        Args:
            num_epochs(int):    number of epochs to train for.
            batch_size(int):    batch size.
            callbacks(list):    list of callback functions
    
        Returns:
            None
        """

        # create a dataloader
        data_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        # train model for 'num_epochs'
        for self.epoch_idx in range(num_epochs):

            # operations 'before' every epoch
            self.on_epoch_begin()

            # training loop
            for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(data_loader)):
                
                # forward pass
                self.optimizer.zero_grad()
                batch_predictions = self.net(batch_images)

                # classification loss + regression loss
                self.loss['train_cls'] = self.class_criterion(batch_predictions[0], batch_labels[0])
                self.loss['train_reg'] = self.box_criterion(batch_predictions[1], batch_labels[1])
                batch_loss = self.cls_loss_weight*self.loss['train_cls'] + self.reg_loss_weight*self.loss['train_reg']

                # backward pass
                self.batch_loss.backward()
                self.optimizer.step()

                # sum batch losses
                self.loss['train'] += batch_loss

            # operations 'after' every epoch
            self.on_epoch_end()


    def evaluate(self, dataset):
        """Function to evaluate the detection network.

        Args:
            dataset(torch.utils.data.Dataset): dataset to evaluate
        """

        # create a dataloader
        data_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        # set network to eval mode
        self.net.eval()

        # run inference on test set
        for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(data_loader)):
            # forward pass
            batch_predictions = self.net(batch_images)

            # TODO: get precision, recall and compute hmean

        # log metrics information for every epoch - precision, recall, hmean
        self.metrics.append({})
        self.metrics[self.epoch_idx]['precision'] = precision
        self.metrics[self.epoch_idx]['recall'] = recall
        self.metrics[self.epoch_idx]['hmean'] = hmean

    def plot_loss(self):
        """Function to represent training/test loss"""
        pass

    def on_epoch_begin(self):
        
        # set network to train mode
        self.net.train()


    def on_epoch_end(self):
        """Function to print loss, calculate val/test metrics, save model, lr annealing etc."""

        print('[/] training - classification loss: {}, regression loss: {}, total loss: {}'.format(self.loss['train_cls'], 
                    self.loss['train_reg'], self.loss['train']))

        # compute validation and test metrics for epoch 'self.epoch_idx'
        self.evaluate(self.test_dataset)

        # save model after every epoch based on a 'criteria'
        self.save_model(criteria='best_model', metric='hmean')

        # adjust learning rate if required
        self.adjust_lr()

    def save_model(self, criteria='best_model', metric='hmean'):
        """Function to save model to disk.

        model filename: datasetname_epochnum.pth
        
        Args:
            criteria: criteria based on which the model will be saved.
            metric: save model if metric improves.
        
        """

        model_filename = '{}_{}.pth'.format(self.dataset_name, self.epoch_idx) ## TODO dataset_name

        # save only best model i.e if 'metric' improves
        if criteria == 'best_model':
            if self.metrics[self.epoch_idx][metric] > self.metrics[self.epoch_idx-1][metric]:
                torch.save(self.net, model_filename)

