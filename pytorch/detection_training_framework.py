from torch.utils.data import DataLoader

from tqdm import tqdm

"""An object detection training framework for PyTorch."""

class DetectionTrainer:

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
            net(torch.nn.Module):   Network architecture.
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
        self.train_loss = -1.
        self.train_cls_loss = -1.
        self.train_reg_loss = -1.
        self.test_loss = -1.
        self.test_cls_loss = -1.
        self.test_reg_loss = -1.
        self.cls_loss_weight = 1.
        self.reg_loss_weight = 1.


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
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        # training loop
        for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(self.data_loader)):
            
            # forward pass
            self.optimizer.zero_grad()
            batch_predictions = self.net(batch_images)

            # classification loss + regression loss
            self.cls_loss = self.class_criterion(batch_predictions[0], batch_labels[0])
            self.reg_loss = self.box_criterion(batch_predictions[1], batch_labels[1])
            self.train_loss = self.cls_loss_weight*self.cls_loss + self.reg_loss_weight*self.reg_loss

            # backward pass
            self.train_loss.backward()
            self.optimizer.step()

            # plot loss
            self.plot_loss()


    def evaluate(self):
        pass


    def plot_loss(self):
        """Function to represent training/test loss"""

        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass