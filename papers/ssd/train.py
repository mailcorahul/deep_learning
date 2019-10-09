import torch.optim as optim
import torch.nn as nn

from .bin.detection_trainer import DetectionTrainer
from .datasets import PascalVOC2007
from .nets import ssd

if __name__ == '__main__':

    # define datasets, network architecture, optimizer and criterions.
    train_dataset, test_dataset = PascalVOC2007(download=True)
    ssd_300 = ssd.SSD300()
    adam_optim = optim.Adam(ssd_300.parameters(), lr=0.01)
    class_criterion = nn.BCEWithLogitsLoss()
    box_criterion = nn.MSELoss()
    
    detection_trainer = DetectionTrainer(
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        net=ssd_300, 
                        optimizer=adam_optim,
                        class_criterion=class_criterion,
                        box_criterion=box_criterion
                        )