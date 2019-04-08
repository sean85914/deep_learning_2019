'''
  DeepConvNet pytorch network
  @param
    f1: conv1 output channel
    f2: conv2 output channel
    f3: conv3 output channel
    f4: conv4 output channel
    fout: conv5 output channel
    k1: conv1 kernel size
    k2: conv2 kernel size
    k3: conv3 kernel size
    k4: conv4 kernel size
    k5: conv5 kernel size
    do: dropout probability
'''

# Torch
import torch
from torchvision import datasets, transforms
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class DeepConvNet(torch.nn.Module):
    def __init__(self, activation, f1=25, f2=25, f3=50, f4 = 100, fout=200, 
                 k1=5, k2=2, k3=5, k4=5, k5=5, do=0.5):
        super(DeepConvNet, self).__init__()
        # Activation:
        #   0 -> ELU
        #   1 -> ReLU
        #   2 -> LeakyReLU
        conv1 = nn.Conv2d(1, f1, kernel_size = (1, k1))
        conv2 = nn.Conv2d(f1, f2, kernel_size = (k2, 1))
        conv3 = nn.Conv2d(f2, f3, kernel_size = (1, k3))
        conv4 = nn.Conv2d(f3, f4, kernel_size = (1, k4))
        conv5 = nn.Conv2d(f4, fout, kernel_size = (1, k5))
        dropout = nn.Dropout(p=do)
        maxpool = nn.MaxPool2d(kernel_size = (1, 2))
        batch_norm_1 = nn.BatchNorm2d(f1)
        batch_norm_2 = nn.BatchNorm2d(f3)
        batch_norm_3 = nn.BatchNorm2d(f4)
        batch_norm_4 = nn.BatchNorm2d(fout)
        if activation == 0:
            activation = nn.ELU()
        elif activation == 1:
            activation = nn.ReLU()
        else:
            activation = nn.LeakyReLU()
        
        self.net1 = nn.Sequential(
            conv1, conv2, batch_norm_1, activation, maxpool, dropout)
        self.net2 = nn.Sequential(
            conv3, batch_norm_2, activation, maxpool, dropout)
        self.net3 = nn.Sequential(
            conv4, batch_norm_3, activation, maxpool, dropout)
        self.net4 = nn.Sequential(
            conv5, batch_norm_4, activation, maxpool, dropout)
        self.classify = nn.Sequential(nn.Linear(fout*37, 2))
        
    def forward(self, x):
        res = self.net1(x)
        res = self.net2(res)
        res = self.net3(res)
        res = self.net4(res)
        res = res.view(res.size(0), -1) # Flatten
        res = self.classify(res)
        return res
