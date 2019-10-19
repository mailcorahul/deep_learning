import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_data = datasets.MNIST('', train=True, transform=transforms.Compose(
    [transforms.ToTensor()]), download=True)

test_data = datasets.MNIST('', train=False, transform=transforms.Compose(
    [transforms.ToTensor()]), download=True)

train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc = nn.Linear(16*5*5, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc(x))
        return F.softmax(self.output(x), dim=1)

net = Network()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

for epoch in range(3):
    running_loss = 0.0
    for i, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = net(X)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 5 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
