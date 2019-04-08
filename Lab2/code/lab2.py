import time
import numpy as np
from argparse import ArgumentParser
# Visualization
from matplotlib import pyplot as plt
# Dataloader
from dataloader import read_bci_data
# Torch
import torch
from torchvision import datasets, transforms
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
# EEG
from eeg import EEGNet
# DeepConvNet
from dcn import DeepConvNet

# argument parser
parser = ArgumentParser(description="Deep Learning and Practice, lab 2 code")
parser.add_argument("--model_type", help="type of model, 0 for EEGNet and 1 for DeepConvNet, default is 0",\
                   default=0)
args = parser.parse_args()

# Hyper parameters
batch_size = 120
num_workers = 0
epoch = 300
global lr
lr = 5e-3
device = 'cuda'

# Read data and convert to torch tensor
train_data, train_label, test_data, test_label = read_bci_data()
train_data = torch.from_numpy(train_data).to(device=device, dtype=torch.float)
train_label = torch.from_numpy(train_label).to(device=device, dtype=torch.float)
test_data = torch.from_numpy(test_data).to(device=device, dtype=torch.float)
test_label = torch.from_numpy(test_label).to(device=device, dtype=torch.float)

# Convert to dataset
train_tensor = Data.TensorDataset(train_data, train_label)
test_tensor = Data.TensorDataset(test_data, test_label)

train_loader = Data.DataLoader(
    dataset = train_tensor,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers
)
test_loader = Data.DataLoader(
    dataset = test_tensor,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers
)

# 0 < epoch < 500: 0.005
# 500 < epoch < 1000: 0.0025
# 1000 < epoch < 1500: 0.00125
# 1500 < epoch < 2000: 0.000625

def adjust_learning_rate(optimizer, epoch):
    if epoch > 500 and epoch < 1000:
        global lr
        lr = 0.0025
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch > 1000 and epoch < 1500:
        global lr
        lr = 0.00125
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch > 1500 and epoch < 2000:
        global lr
        lr = 0.000625
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
# Train and test
def train(model, optimizer, Loss,  epoch):
    model.train()
    loss_now = None
    for param_group in optimizer.param_groups:
            temp = param_group['lr'] 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        target = target.long()
        
        optimizer.zero_grad()
        out = model(data)
        loss = Loss(out, target)
        loss.backward()
        optimizer.step()

        loss_now = loss.item()
    print('Training epoch: {}\t | lr: {}\t | Loss: {:.6f}'.format(epoch, temp, loss_now))
        
        
        
def test(model, epoch, activation_type, phase="Test"):
    if activation_type == 0:
        activation_str = "ELU"
    elif activation_type == 1:
        activation_str = "ReLU"
    else:
        activation_str = "LeakyReLU"
        
    if phase == "Test":
        data_loader = test_loader
        num = len(test_label)
    else: # train
        data_loader = train_loader
        num = len(train_label)
    correct_num = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        target = target.long()
        out = model(data)
        _, predicted = torch.max(out, 1)
        for i in range(len(target)):
            if predicted.cpu()[i] == target.cpu()[i]:
                correct_num += 1
    acc = correct_num/float(num)
    print("{} with activation type {} epoch: {}, accuracy: {:.2f}".format
          (phase, activation_str, epoch, acc))
    return acc

def EEG_train_test():
	f1 = 16
	f2 = 32
	fout = 32
	k1 = 51
	k2 = 2
	k3 = 16
	do = 0.25

	fig, ax = plt.subplots(figsize=(8, 6))
	ACC = []
	for activation_type in range(3):
		model = EEGNet(activation_type, f1, f2, fout)
		model.to(device)
		# Define optimizer and loss function
		Loss = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr = lr)
		train_acc = []
		test_acc = []
		acc_max = 0.8 # Only save result higher then 0.8
		for i in range(1, epoch+1):
			train(model, optimizer, Loss, i)
			print("-----------------------------------------------------------")
			acc = test(model, i, activation_type, "Train")
			train_acc.append(acc)
			acc = test(model, i, activation_type, "Test")
			test_acc.append(acc)
			print("-----------------------------------------------------------")
			if activation_type == 0:
				activation_str = "ELU"
			elif activation_type == 1:
				activation_str = "ReLU"
			else:
				activation_str = "LeakyReLU"
			if acc > acc_max:
				path = "model/eeg_activation_" + activation_str + "_iteration_{}_acc_{:.3f}_lr_{}.pth".format(i, acc, lr)
				torch.save(model, path)
				acc_max = acc
			train_str = "Training acc. with " + activation_str
			test_str = "Testing acc. with " + activation_str
		ACC.append(train_acc)
		ACC.append(test_acc)
		ax.plot(train_acc, label=train_str)
		ax.plot(test_acc, label=test_str)    
	ax.legend(loc="best")
	plt.xlabel("Epoch")
	plt.ylabel("Accuacy (%)")
	plt.title("result_eeg")
	plt.savefig("result_eeg.png")
	plt.show()
	
def DeepConvNet_train_test():
	fig, ax = plt.subplots(figsize=(8, 6))
	ACC_ = []

	for activation_type in range(3):
		model = DeepConvNet(activation_type, fout=150, k1=11, k2=2, k3=11, k4=11, k5=12)
		model.to(device)
		# Define optimizer and loss function
		Loss = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr = lr)
		train_acc = []
		test_acc = []
		acc_max = 0.8 # Only save result higher then 0.8
		for i in range(1, epoch+1):
			# adjust learning rate
			adjust_learning_rate(optimizer, i)
			train(model, optimizer, Loss, i)
			print("-----------------------------------------------------------")
			acc = test(model, i, activation_type, "Train")
			train_acc.append(acc)
			acc = test(model, i, activation_type, "Test")
			test_acc.append(acc)
			print("-----------------------------------------------------------")
			if activation_type == 0:
				activation_str = "ELU"
			elif activation_type == 1:
				activation_str = "ReLU"
			else:
				activation_str = "LeakyReLU"
			train_str = "Training acc. with " + activation_str
			test_str = "Testing acc. with " + activation_str
			if acc > acc_max:
				path = "model/dcn_activation_" + activation_str + "_iteration_{}_acc_{:.3f}_lr_{}.pth".format(
				i, acc, lr)
				torch.save(model, path)
				acc_max = acc
			print("acc_max is now: {}".format(acc_max))
		ACC_.append(train_acc)
		ACC_.append(test_acc)
		ax.plot(train_acc, label=train_str)
		ax.plot(test_acc, label=test_str)    
	ax.legend(loc="best")
	plt.xlabel("Epoch")
	plt.ylabel("Accuacy (%)")
	plt.title("result_dcn")
	plt.savefig("result_dcn.png")
	plt.show()
	
if __name__== "__main__":
	if args.model_type == 0:
		EEG_train_test()
	else:
		DeepConvNet_train_test()
