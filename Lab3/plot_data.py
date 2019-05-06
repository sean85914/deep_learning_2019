'''
  Plot data that recorded training and testing accuracy
  Editor: Sean Lu
  Last Edited: 4/17
'''
from matplotlib import pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(description="Plot curves with given data and title")
parser.add_argument("--fin", help="file input")
parser.add_argument("--t", "--title", help="title of plot")
parser.add_argument("--fo", help="file output")
args = parser.parse_args()
# File encode in 
# train_acc1 train_acc2 train_acc3 ...
# test_acc1  test_acc2  test_acc3  ...
title = str(args.t)
fig_name = str(args.fo) + "png"
f = open(str(args.fin))
train_str = f.readline()
test_str = f.readline()
train_str = train_str.split(" ")
test_str  = test_str.split(" ")
train_list = []
test_list = []

for x in train_str:
    train_list.append(float(x))
for x in test_str:
    test_list.append(float(x))
    
plt.plot(train_list, label="Training Accuracy")
plt.plot(test_list, label="Testing Accuracy")
plt.title(title)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend(loc="best")
plt.savefig(title, format="png")

