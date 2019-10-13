import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)
    return torch.div(exp_x, sum_x)

def log_softmax(x):
    return torch.log(torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True))

def CrossEntropyLoss(outputs, targets):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs = log_softmax(outputs)
    outputs = outputs[:, targets]

    return - torch.sum(outputs)/num_examples

input_dim = 784 # 28x28 FashionMNIST data
output_dim = 10

w_init = np.random.normal(scale=0.05, size=(input_dim,output_dim))
w_init = torch.tensor(w_init, requires_grad=True)
b = torch.zeros(output_dim).double()

def my_model(x):
    bs = x.shape[0]
    return torch.matmul(x.reshape(bs, input_dim), w_init) + b

criterion = nn.CrossEntropyLoss()
trn_fashion_dl = [(np.random.randn(1, 28, 28), np.array([1]))]

for X, y in trn_fashion_dl:
    X = torch.tensor(X)
    y = torch.tensor(y)
    outputs = my_model(X)
    #my_outputs = softmax(outputs)
    print(outputs, outputs.size())
    my_ce = CrossEntropyLoss(outputs, y)
    pytorch_ce = criterion(outputs, y)

    print('my custom cross entropy: {:.6f}\npytorch cross entroopy: {:.6f}'.format(my_ce.item(), pytorch_ce.item()))
    break 