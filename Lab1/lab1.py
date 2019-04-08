'''
  DLP lab1 code

  Author: Sean Lu
  Last edited: 3/21, 2019

  How to use? 

  $ python3 lab1.py --model_type [model_type] --epoch [epoch] --lr [lr] --loss_type [loss_type]
  For example, training Separable dataset with 0.01 learning rate and 10000 epoch over cross entropy
  $ python3 lab1.py --model_type 0 --epoch 10000 --lr 0.01 --loss_type 0
  For more detail, use
  $ python3 lab1.py -h
'''

# import libraries
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib import interactive
# argument parser
parser = ArgumentParser(description="Deep Learning and Practice, homework 2 code")
parser.add_argument("--model_type", help="type of model, 0 for Separable and 1 for XOR, default is 0",\
                   default=0)
parser.add_argument("--epoch", help="number of iteration, default is 10000", default=10000)
parser.add_argument("--lr", help="learning_rate, default is 0.01", default=1e-2)
parser.add_argument("--loss_type", help="type of loss, 0 for cross entropy and 1 for MSE, default is 0", default=0)
parser.add_argument("--output", "-o", help="output file name as string, default is None, i.e., no format will be saved", default="None")

args = parser.parse_args()

# Function to generate training data
'''
  Linear separable dataset generator
  @param
    n: number of data
'''
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)
'''
  XOR dataset generator
  @param
    None
'''
def generate_XOR_easy():
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(21, 1)
# Generate training data and save to memories
x_train, y_train = generate_linear(n=100)
x_train_1 , y_train_1 = generate_XOR_easy()
# Active function and its derivative
'''
  Sigmoid function
'''
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
'''
  Derivative of sigmoid
'''
def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)
# Display funtion
'''
  @param
    x: data
    y: corresponding data groundtruth
    y_pred: if given, plot the compare graph with this prediction result,
            the result is in range [0, 1], so value with greater than 0.5 
            will be recognized as Class 1 and less than 0.5 as Class 0
'''
def show_result(x, y, y_pred = None):
    plt.figure(2)
    if y_pred is not None:
        plt.subplot(1, 2, 1)
        plt.title("Ground truth", fontsize = 18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
    else:
        plt.title("Ground truth", fontsize = 18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
    if y_pred is not None:
        plt.subplot(1, 2, 2)
        plt.title("Predict result", fontsize = 18)
        for i in range(x.shape[0]):
            # If prediction smaller than 0.5, classify as class 0
            if y_pred[i] < 0.5:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
## NN Architecture
## Input(2) -> FC1(2) -> FC(2) -> OUT(1)
'''
  Forward propagation
  @param
    x: input with 2 channel
    W: weight
'''
def forward(x, W):
    w1, w2, w3 = W
    # Forward
    z1 = np.dot(x, w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2)
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3)
    y_pred = sigmoid(z3)
    return y_pred
# Random initial weight and return 
def init_weight():
    w1 = np.random.randn(2, 2)
    w2 = np.random.randn(2, 2)
    w3 = np.random.randn(2, 1)
    print("w1 initial: \n", w1)
    print("w2 initial: \n", w2)
    print("w3 initial: \n", w3)
    return [w1, w2, w3]
# Save result in memories
def save_result():
    global w1, w2, w3
    W = [w1, w2, w3]
    print('Result saved.')
    print(W)
    return W
###################################################################################
# Train
loss_sum = 0
if args.loss_type == 0:
    loss_type = 'Cross Entropy'
else:
    loss_type = 'MSE'
'''
  x: training data
  y: corresponding groundtruth
  i: epoch now
  j: index of data now
  model: indicate which model, 0 for linear separable and 1 for XOR
'''
def train(x, y, loss_list = None,i = None, j = None, model = 0):
    global w1, w2, w3
    # Forward
    z1 = np.dot(x, w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2)
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3)
    y_pred = sigmoid(z3)   
    # Loss
    if loss_type == 'Cross Entropy':
        loss = -y * np.log(y_pred) - (1-y) * np.log(1-y_pred)
        loss_grad = (y_pred - y)/ (y_pred * (1-y_pred))
    if loss_type == 'MSE':
        loss = (y - y_pred)**2 /2.
        loss_grad = y_pred - y 
    # Initial grad
    w3_grad = np.zeros((2, 1))
    w2_grad = np.zeros((2, 2))
    w1_grad = np.zeros((2, 2))
    # w3
    w3_grad[0] = loss_grad * derivative_sigmoid(y_pred) * a2[0]
    w3_grad[1] = loss_grad * derivative_sigmoid(y_pred) * a2[1]
    
    way_1 = loss_grad * derivative_sigmoid(y_pred) * w3[0] * derivative_sigmoid(a2[0])
    way_2 = loss_grad * derivative_sigmoid(y_pred) * w3[1] * derivative_sigmoid(a2[1])
    # w2
    w2_grad[0][0] = way_1 * a1[0]
    w2_grad[0][1] = way_2 * a1[0]
    w2_grad[1][0] = way_1 * a1[1]
    w2_grad[1][1] = way_2 * a1[1]
    
    way_3 = way_1 * w2[0][0] * derivative_sigmoid(a1[0]) + \
            way_2 * w2[0][1] * derivative_sigmoid(a1[0])
    way_4 = way_1 * w2[1][0] * derivative_sigmoid(a1[1]) + \
            way_2 * w2[1][1] * derivative_sigmoid(a1[1])
    # w1
    w1_grad[0][0] = way_3 * x[0]
    w1_grad[0][1] = way_4 * x[0]
    w1_grad[1][0] = way_3 * x[1]
    w1_grad[1][1] = way_4 * x[1]
    # Update weight
    w1 -= learning_rate * w1_grad
    w2 -= learning_rate * w2_grad
    w3 -= learning_rate * w3_grad
    global loss_sum
    loss_sum += abs(loss[0])
    # Define size of dataset and output frequency
    if model == 0:
        NUM = 100.
        FREQ = 500
    else:
        NUM = 21.
        FREQ = 10000  
    if loss_list is not None:
        if j == NUM - 1:
            loss_list.append(loss_sum/NUM /(i+1))
    if((i+1)%FREQ == 0 and j == NUM - 1):
        print("Epoch: {:5d}, loss: {}".format(i+1, loss_sum/NUM /(i+1)))
        loss_sum = 0
###################################################################################  
# Start to train
epoch = int(args.epoch)
learning_rate = float(args.lr)
w1, w2, w3 = init_weight()
loss_list = []
if args.model_type == 0: data_len = 100 
else: data_len = 21 
for i in range(epoch):
    for j in range(data_len):
        if args.model_type == 0:
            train(x_train[j], y_train[j], loss_list, i, j)
        else:
            train(x_train_1[j], y_train_1[j], loss_list, i, j, args.model_type)
        
print("Training process terminated.")
# Plot loss curve
plt.figure(1)
plt.plot(loss_list)
plt.title("Loss curve with lr={}".format(learning_rate))
# Save result
W = save_result()
if args.model_type == 0:
    show_result(x_train, y_train, forward(x_train, W))
else:
    show_result(x_train_1, y_train_1, forward(x_train_1, W))
plt.show()

if args.output is not "None":
	f = open(args.output, 'w+')
	for i in range(len(W)):
		f.write(str(W[i]))
	f.close()
