'''
  Test given model against traing set and testing set and print both accuracies
  Editor: Sean Lu
  Last Edited: 4/17
'''
import pandas as pd
from torch.utils import data
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import torch.utils.data as Data
import torchvision.models as models
import numpy as np
import os
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import time
import copy
import matplotlib.pyplot as plt
from dataset import *
from test import test_model

num_classes = 5

    
device = 'cuda'
'''
transform_train = transforms.Compose([transforms.ToPILImage(), # First convert to PIL image
                                      transforms.RandomHorizontalFlip(p=0.5), # data augmentation
                                      transforms.RandomVerticalFlip(p=0.5), # data augmentation
                                      transforms.RandomRotation((0, 360)), # data augmentation
                                      transforms.ToTensor(), # Then convert to tensor for training
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # standardize
                                      ])
'''
transform_test  = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

rpl_train = RetinopathyLoader("data", "train", transform_test)
rpl_test  = RetinopathyLoader("data",  "test", transform_test)
num_workers = 4
batch_size = 8
train_loader = Data.DataLoader(rpl_train, batch_size, shuffle=False, num_workers=num_workers)
test_loader = Data.DataLoader(rpl_test, batch_size, shuffle=False, num_workers=num_workers)

train_acc = []
test_acc = []
train_len = len(rpl_train)
test_len = len(rpl_test)

'''epochs = 20
for i in range(20, epochs+1):
    print("At epoch: {}".format(i))
    path = "model/temp/resnet_50_epoch_{}_batch_15_lr_0.001_pretrained_acc_0.000.pth".format(i)
    model = torch.load(path)
    acc = test_model(model, train_loader, train_len, "train") # Test model using training dataset
    acc = round(acc, 3)
    train_acc.append(acc)
    acc = test_model(model, test_loader, test_len, "test") # Test model using testing dataset
    acc = round(acc, 3)
    test_acc.append(acc)
    
    
print(train_acc)
print(test_acc)'''
# Test greatest one
model = torch.load("model/temp/resnet_50_epoch_10_batch_15_lr_0.001_pretrained_acc_0.822.pth")
acc = test_model(model, train_loader, train_len, "train")
print("Testing against training set {:.3f}".format(acc))
acc = test_model(model, test_loader, test_len, "test")
print("Testing against testing set {:.3f}".format(acc))
