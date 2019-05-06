#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt

# Constants
BIN_DIM = 8
INPUT_DIM = 2
HIDDEN_DIM = 16
OUT_DIM = 1
ALPHA = 0.1 # Learning rate
ITER = 20000
TEST_MAX = 100
LOG_ITER = ITER // 20 # Print information, 1000

def dec2bin(x):
    if x<0:
        b = bin(x & 0b11111111)[2:]
    else:
        b = bin(x)[2:].zfill(8)
    b = "{} {} {} {} {} {} {} {}".format(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7])
    b_arr = np.fromstring(b, sep=' ', dtype=int)
    return b_arr

largest = pow(2, BIN_DIM-1) # 128
decimal = np.array([range(-largest, largest)]).astype(int).T # Build decimal map, range [0, 255]
binary = np.zeros((largest*2, BIN_DIM))
for i in range(len(decimal)):
    binary[i][:] = dec2bin(decimal[i][0])

# Activation function, using sigmoid
def sigmoid(x): #[0, 1]
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(out):
    return out * (1 - out)
# Convert binary array to decimal integer
# Have to consider 2's complement
def bin2dec(b):
    out = 0
    is_negative = 1
    if b[0] == 1:
        is_negative = -1
        for i in range(BIN_DIM): # Take inverse to all values
            if b[i] == 1:
                b[i] = 0
            else: b[i] = 1
    for i, x in enumerate(b[::-1]):
        if i == 7: continue # just for sign
        out += x * pow(2, i)
    if is_negative == -1:
        out = -out -1
    return int(out)

# Initial weights by normal distribution
# Work better
U = np.random.normal(0, 1, [HIDDEN_DIM, INPUT_DIM]) # input -> hidden, 16*2
W = np.random.normal(0, 1, [HIDDEN_DIM, HIDDEN_DIM]) # hidden -> hidden, 16*16
V = np.random.normal(0, 1, [OUT_DIM, HIDDEN_DIM]) # hidden -> output, 1*16 
'''
U = np.random.uniform(-5, 5, (HIDDEN_DIM, INPUT_DIM)) # input -> hidden, 16*2
W = np.random.uniform(-5, 5, (HIDDEN_DIM, HIDDEN_DIM)) # hidden -> hidden, 16*16
V = np.random.uniform(-5, 5, (OUT_DIM, HIDDEN_DIM)) # hidden -> output, 1*16
'''
# Initial gradient
dU = np.zeros_like(U)
dW = np.zeros_like(W)
dV = np.zeros_like(V)

acc_list = list() # Accuracy list for ploting
accuracy = 0
# Train
for i in range(1, ITER+1):
    # Random take a, b
    a_dec = np.random.randint(-largest/2, largest/2) # [-64, 63]
    b_dec = np.random.randint(-largest/2, largest/2) # [-64, 63]
    c_dec = a_dec + b_dec # c = a + b
    # In binary
    a_bin = dec2bin(a_dec)
    b_bin = dec2bin(b_dec)
    c_bin = dec2bin(c_dec)
    # Predict placeholder
    pred = np.zeros_like(c_bin)
    # Temp placeholder
    hidden = np.zeros([BIN_DIM+1, HIDDEN_DIM]) # hidden layer (h), 9*16
    predict = np.zeros([BIN_DIM]) # predict (y_pred), 8
    ground_truth = np.zeros([BIN_DIM]) # ground truth (y), 8
    H = np.zeros([BIN_DIM, HIDDEN_DIM, HIDDEN_DIM]) # dh/da, 8*16*16
    overall_err = 0

    # forward propagation
    for pos in range(BIN_DIM)[::-1]: # 7 6 5 4 3 2 1 0
        X = np.array([[a_bin[pos]], [b_bin[pos]]], dtype=np.float64) # shape = (2, 1), input
        Y = np.array([[c_bin[pos]]], dtype=np.float64) # shape = (1, 1), ground truth
        ground_truth[BIN_DIM-pos-1] = Y
        a = np.dot(W, hidden[BIN_DIM-pos-1].reshape((HIDDEN_DIM, OUT_DIM)))+np.dot(U, X) # a(t) = W*h(t-1)+U*x(t), (16, 1)
        h = sigmoid(a) # h(t) = sigmoid(a(t)), (16, 1)
        hidden[BIN_DIM-pos][:] = h.reshape((HIDDEN_DIM)) # 1 2 3 4 5 6 7 8
        H[BIN_DIM-pos-1] = np.diag(deriv_sigmoid(h).reshape(HIDDEN_DIM)) # dh/da, 0 1 2 3 4 5 6 7
        o = np.dot(V, h) # o(t) = V*h(t), (1, 1)
        y_pred = sigmoid(o) # y_pred(t) = sigmoid(o(t)), (1, 1)
        predict[BIN_DIM-pos-1] = y_pred # 0 1 2 3 4 5 6 7
        pred[pos] = np.round(y_pred) # Round to get predict result (either 0 or 1)
        # Cross entropy
        output_err = -Y*np.log(y_pred) - (1-Y)*np.log(1-y_pred)
        overall_err += output_err
    #######################################################################################
    #######################################################################################
    #                                                                                     #
    #                            backprogagation through time                             #
    #                                                                                     #
    #######################################################################################
    #######################################################################################
    for pos in range(BIN_DIM)[::-1]: # 7 6 5 4 3 2 1 0
        X = np.array([[a_bin[pos]], [b_bin[pos]]], dtype=np.float64) # shape = (2, 1), input
        pre = predict[BIN_DIM-pos-1] # 0 1 2 3 4 5 6 7
        gt = ground_truth[BIN_DIM-pos-1] # 0 1 2 3 4 5 6 7
        dLdo = pre - gt # dL/do = dL/d(y_pred) * d(y_pred)/do = y_pred - y
        dLdh = np.dot(V.T, dLdo) # common term: V.T*(dL/do(t))
        dV += np.dot(dLdo, hidden[BIN_DIM-pos].T) # dL/dv = sum(dL/do(t))*h(t).T, 1 2 3 4 5 6 7 8
        if (pos != 0): # if not last term, has to look ahead one term
            pre = predict[BIN_DIM-pos] # +1
            gt = ground_truth[BIN_DIM-pos] # +1
            dLdo = pre - gt
            # dLdh += W.T*H(t+1)*V.T*(dL/dh(t+1)) (so becomes W.T*H(t+1)*V.T*(dL/dh(t+1)) + V.T*(dL/do(t)))
            dLdh += np.dot(np.dot(W.T, H[BIN_DIM-pos]), np.dot(V.T, dLdo)) 
        dU += np.dot(np.dot(H[BIN_DIM-pos-1], dLdh), X.T) # dL/dU = sum(H(t)*(dL/dh(t))*x(t).T)
        dW += np.dot(np.dot(H[BIN_DIM-pos-1], dLdh), hidden[BIN_DIM-pos-1].reshape((HIDDEN_DIM, 1)).T) # dL/dW = sum(H(t)*(dL/dh(t))*h(t-1).T)
    # Update weights
    U -= ALPHA*dU
    W -= ALPHA*dW
    V -= ALPHA*dV
    # Zero grad
    dU *= 0
    dW *= 0
    dV *= 0
    
    if bin2dec(pred) == c_dec:
        if bin2dec(pred)+c_dec != -1:
            accuracy += 1


    if (i % LOG_ITER == 0):
        print("*"*30)
        print("Iteration: {}".format(i))
        print("Error: {}".format(overall_err[0][0]))
        print("a:       {}".format(a_bin))
        print("b:       {}".format(b_bin))
        print("Predict: {}".format(pred))
        print("True:    {}".format(c_bin))
        print(a_dec, "+", b_dec, "=", bin2dec(pred))
        if bin2dec(pred) == c_dec:
            if bin2dec(pred)+c_dec != -1:
                print("Correct!")
        else: print("Incorrect@@")
        acc_list.append(accuracy/float(LOG_ITER))
        print("Accuracy: {}%".format(accuracy/float(LOG_ITER)*100.))
        accuracy = 0
    

plt.plot(acc_list)
plt.title("RNN Training Accuracy Curve", fontsize=18)
plt.xlabel("Iteration/{:d}".format(LOG_ITER), fontsize=18)
plt.ylabel("Accuracy(%)", fontsize=18)
plt.show()

# Forward part, just copy from above
def forward(a_bin, b_bin, U=U,V=V,W=W):
    pred = np.zeros([BIN_DIM])
    hidden = np.zeros([BIN_DIM+1, HIDDEN_DIM]) # hidden layer, 9*16
    for pos in range(BIN_DIM)[::-1]: # 7 6 5 4 3 2 1 0
        X = np.array([[a_bin[pos]], [b_bin[pos]]], dtype=np.float64) # shape = (2, 1), input
        Y = np.array([[c_bin[pos]]], dtype=np.float64) # shape = (1, 1), ground truth
        
        a = np.dot(W, hidden[BIN_DIM-pos-1].reshape((HIDDEN_DIM, OUT_DIM)))+np.dot(U, X) # a(t) = W*h(t-1)+U*x(t), (16, 1)
        h = sigmoid(a) # h(t) = sigmoid(a(t)), (16, 1)
        o = np.dot(V, h) # o(t) = V*h(t), (1, 1)
        y_pred = sigmoid(o) # y_pred(t) = sigmoid(o(t)), (1, 1)
        pred[pos] = np.round(y_pred) # Round to get predict result (either 0 or 1)
        hidden[BIN_DIM-pos][:] = h.reshape((HIDDEN_DIM))
    return bin2dec(pred)
# Testing
print("Testing... ")
acc = 0
for i in range(TEST_MAX):
    a_dec = np.random.randint(-largest/2, largest/2) # [-64, 63]
    b_dec = np.random.randint(-largest/2, largest/2) # [-64, 63]
    c_dec = a_dec + b_dec # c = a + b
    # In binary
    a_bin = dec2bin(a_dec)
    b_bin = dec2bin(b_dec)
    c_bin = dec2bin(c_dec)
    if forward(a_bin, b_bin) == c_dec: 
        acc+=1
    print("*"*30)
    print("a_bin:   {} ({})".format(a_bin, a_dec))
    print("b_bin:   {} ({})".format(b_bin, b_dec))
    print("Predict: {} ({})".format(dec2bin(forward(a_bin, b_bin)), forward(a_bin, b_bin)))
    if forward(a_bin, b_bin) == c_dec: print("Correct")
    else: print("Incorrect@@")

print(acc/float(TEST_MAX)*100)
