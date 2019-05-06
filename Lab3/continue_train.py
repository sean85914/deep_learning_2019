'''
  Continue to train model that no good enough
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
from argparse import ArgumentParser
from ResNet import *
from dataloader import *

# argument parser
parser = ArgumentParser(description="Continue to train with given model")
parser.add_argument("--model", help="model trained")
parser.add_argument("--prefix", help="Save file name prefix")
parser.add_argument("--batch_size", "--bs", help="Batch size, default is 4", default=4)
parser.add_argument("--learning_rate", "--lr", help="Learning rate, default is 1e-3", default=0.001)
parser.add_argument("--epochs", help="Epoch, default is 10", default=10)
args = parser.parse_args()

batch_size = int(args.batch_size)
lr = float(args.learning_rate)
epochs = int(args.epochs)
model_path = str(args.model)
prefix = str(args.prefix)

def train_model(model, dataloader, criterion, optimizer, idx):
    since = time.time()
    model.train() # Set model to training mode
    loss_now = None
    for batch_idx, (data, target) in enumerate(dataloader):
        print("[{}/{}]...\r".format(batch_idx, len(dataloader)), end="")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        loss_now = loss.item()
    cost_time = time.time()-since
    print('Train epoch: {}\t | Loss: {:.6f} \t | Cost time: {:} m: {} s'.
         format(idx, loss_now, int(cost_time/60), int(cost_time%60)))
         
def save_model(model, acc, idx, batch):
    path = "model/{}_epoch_{}_batch_{}_lr_{}_acc_{:.3f}.pth".format(prefix, idx, batch, lr, acc)
    print("Save model: {}".format(path))
    torch.save(model, path)
    
device = 'cuda'
transform_train = transforms.Compose([transforms.ToPILImage(), # First convert to PIL image
                                      transforms.RandomHorizontalFlip(p=0.5), # data augmentation
                                      transforms.RandomVerticalFlip(p=0.5), # data augmentation
                                      transforms.RandomRotation((0, 360)), # data augmentation
                                      transforms.ToTensor(), # Then convert to tensor for training
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # standardize
                                      ])
                                      
rpl_train = RetinopathyLoader("data", "train", transform_train)
num_workers = 4
train_loader = Data.DataLoader(rpl_train, batch_size, shuffle=False, num_workers=num_workers)
model = torch.load(model_path)

Loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), momentum=0.9, 
                      nesterov=True, weight_decay=5e-4, lr = lr)
for i in range(1, epochs+1):
    train_model(model, train_loader, Loss, optimizer, i) # Train model using training dataset
    save_model(model, 0, i, batch_size)

