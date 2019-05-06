'''
  Main Lab3 code
  Editor: Sean Lu
  Last Edited: 4/17
'''
from torch.utils import data
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from ResNet import resnet18, resnet50
from dataset import RetinopathyLoader
from train import train_model
from test import test_model

# argument parser
parser = ArgumentParser(description="Deep Learning and Practice, lab 3 code")
parser.add_argument("--model", help="Option: 0, 1, 2, 3", default=0)
parser.add_argument("--batch_size", "--bs", help="Batch size, default is 4", default=4)
parser.add_argument("--learning_rate", "--lr", help="Learning rate, default is 1e-3", default=0.001)
parser.add_argument("--epochs", help="Epoch, default is 10", default=10)
args = parser.parse_args()

num_classes = 5
def getMyResnet(net_type):
    if net_type == 18:
        model = models.resnet18(pretrained=True)
        fc_in = 512
    else:
        model = models.resnet50(pretrained=True)
        fc_in = 2048
    model.fc = nn.Linear(fc_in, num_classes)
    model.to(device)
    return model
    
def getMyResnetNoPretrained(net_type):
    if net_type == 18:
        model = resnet18()
        model.fc = nn.Linear(512, num_classes)
    else:
        model = resnet50()
        model.fc = nn.Linear(2048, num_classes)
    model.to(device)
    return model
    
def save_model(model, acc, net_type, has_pretrained, idx, batch):
    if has_pretrained:
        pre = "pretrained"
    else:
        pre = "no_pretrained"
    path = "model/resnet_{}_epoch_{}_batch_{}_lr_{}_{}_acc_{:.3f}.pth".format(net_type, idx, batch, lr, pre, acc)
    print("Save model: {}".format(path))
    torch.save(model, path)

'''
  mode = 0: resnet-18 w/  pretrained
  mode = 1: resnet-18 w/o pretrained
  mode = 2: resnet-50 w/  pretrained
  mode = 3: resnet-50 w/o pretrained
'''

mode = int(args.model)
batch_size = int(args.batch_size)
lr = float(args.learning_rate)
epochs = int(args.epochs)

device = 'cuda'
transform_train = transforms.Compose([transforms.ToPILImage(), # First convert to PIL image
                                      transforms.RandomHorizontalFlip(p=0.5), # data augmentation
                                      transforms.RandomVerticalFlip(p=0.5), # data augmentation
                                      transforms.RandomRotation((0, 360)), # data augmentation
                                      transforms.ToTensor(), # Then convert to tensor for training
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # standardize
                                      ])

rpl_train = RetinopathyLoader("data", "train", transform_train)
#rpl_test  = RetinopathyLoader("data",  "test", transform_test)
num_workers = 4
train_loader = Data.DataLoader(rpl_train, batch_size, shuffle=False, num_workers=num_workers)
#test_loader = Data.DataLoader(rpl_test, 1, shuffle=False, num_workers=num_workers)

#f = open("result_data.txt", "a+")

if mode == 0:
    model = getMyResnet(18)
    #f.write("Resnet-18 w/ pretrained\n")
    net_type = 18
    has_pretrained = True
    fig_str = "resnet18_pretrained.jpg"
    fig_title = "ResNet18 w/ Pretrained"
    print("Start training resnet-18 w/ pretrained...")
elif mode == 1:
    model = getMyResnetNoPretrained(18)
    #f.write("Resnet-18 w/o pretrained\n")
    net_type = 18
    has_pretrained = False
    fig_str = "resnet18_pretrained.jpg"
    fig_title = "ResNet18 w/o Pretrained"
    print("Start training resnet-18 w/o pretrained...")
elif mode == 2:
    model = getMyResnet(50)
    #f.write("Resnet-50 w/ pretrained\n")
    net_type = 50
    has_pretrained = True
    fig_str = "resnet50_pretrained.jpg"
    fig_title = "ResNet50 w/ Pretrained"
    print("Start training resnet-50 w/ pretrained...")
else:
    model = getMyResnetNoPretrained(50)
    #f.write("Resnet-50 w/o pretrained\n")
    net_type = 50
    has_pretrained = False
    fig_str = "resnet50_nopretrained.jpg"
    fig_title = "ResNet50 w/o Pretrained"
    print("Start training resnet-50 w/o pretrained...")
    
Loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), momentum=0.9, 
                      nesterov=True, weight_decay=5e-4, lr = lr)
train_acc = []
test_acc = []
for i in range(1, epochs+1):
    train_model(model, train_loader, net_type, has_pretrained, Loss, optimizer, i) # Train model using training dataset
    save_model(model, 0, net_type, has_pretrained, i, batch_size)

