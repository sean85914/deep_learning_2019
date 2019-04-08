'''
  EEGNet pytorch network
  @param
    f1: first conv input channel
    f2: depthwise conv input channel
    fout: sparable conv input channel
    k1: first conv kernel size
    k2: depthwise conv kernel size
    k3: sparable conv kernel size
    do: dropout probability
'''
# Torch
import torch
from torchvision import datasets, transforms
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class EEGNet(torch.nn.Module):
	# Activation:
    #   0 -> ELU
    #   1 -> ReLU
    #   2 -> LeakyReLU
    def __init__(self, activation=0, f1=16, f2=32, fout=32, k1=51, k2=2, k3=15, do=0.25):
        super(EEGNet, self).__init__()
        if activation == 0:
        	activation_f = nn.ELU()
        elif activation == 1:
        	activation_f = nn.ReLU()
        else:
        	activation_f = nn.LeakyReLU()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, k1), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(f1, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=(k2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(f2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation_f, 
            nn.AvgPool2d(kernel_size=(1,4), stride=(1, 4), padding=0),
            nn.Dropout(p=do)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(f2, fout, kernel_size=(1, k3), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(fout, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1, 8), padding=0),
            nn.Dropout(p=do)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=fout*23, out_features=2, bias=True)
        )
    def forward(self, x):
        res = self.firstConv(x)
        res = self.depthwiseConv(res)
        res = self.separableConv(res)
        res = res.view(res.size(0), -1)
        res = self.classify(res)
        return res
