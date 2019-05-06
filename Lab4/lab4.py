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
TEST_MAX = 10000
LOG_ITER = ITER // 20 # Print information, 1000
PLOT_ITER = ITER // 200 # Plot curve, 100

largest = pow(2, BIN_DIM) # 256
decimal = np.array([range(largest)]).astype(np.uint8).T # Build decimal map, range [0, 255]
binary = np.unpackbits(decimal, axis = 1) # Build binary map

#############################################################################################
#############################################################################################
#                                                                                           #
#                                Common Usage Functions                                     #
#                                                                                           #
#############################################################################################
#############################################################################################
# Activation function, using sigmoid
def sigmoid(x): #[0, 1]
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(out):
    return out * (1 - out)
# Convert binary array to decimal integer
def bin2dec(b):
    out = 0
    for i, x in enumerate(b[::-1]):
        out += x * pow(2, i)
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

acc_list = list() # Accuracy list for plotting
loss_list = list() # Loss list for plotting
accuracy = 0

#############################################################################################
#############################################################################################
#                                                                                           #
#                                        Training                                           #
#                                                                                           #
#############################################################################################
#############################################################################################
for i in range(1, ITER+1):
    # Random take a, b
    a_dec = np.random.randint(largest/2)
    b_dec = np.random.randint(largest/2)
    c_dec = a_dec + b_dec # c = a + b
    # In binary
    a_bin = binary[a_dec]
    b_bin = binary[b_dec]
    c_bin = binary[c_dec]
    output_err = 0
    # Predict placeholder
    pred = np.zeros_like(c_bin)
    # Temp placeholder
    hidden = np.zeros([BIN_DIM+1, HIDDEN_DIM]) # hidden layer (h), 9*16
    predict = np.zeros([BIN_DIM]) # predict (y_pred), 8
    ground_truth = np.zeros([BIN_DIM]) # ground truth (y), 8
    H = np.zeros([BIN_DIM, HIDDEN_DIM, HIDDEN_DIM]) # dh/da, 8*16*16
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
        output_err += -Y*np.log(y_pred) - (1-Y)*np.log(1-y_pred)
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
    
    loss_list.append(output_err[0][0])

    if (bin2dec(pred) == c_dec):
        accuracy += 1

    if (i % PLOT_ITER == 0):
        acc = accuracy/float(PLOT_ITER)
        acc_list.append(acc)
            
        accuracy = 0

    if (i % LOG_ITER == 0):
        print("*"*30)
        print("Iteration: {}".format(i))
        print("Error: {}".format(output_err))
        print("Predict: {}".format(pred))
        print("True:    {}".format(c_bin))
        print("Accuracy: {:.0f}%".format(acc_list[-1]*100.))

# Plot accuracy curve
plt.plot(acc_list)
plt.title("RNN Training Accuracy Curve", fontsize=18)
plt.xlabel("Iteration/{:d}".format(PLOT_ITER), fontsize=18)
plt.ylabel("Accuracy(%)", fontsize=18)
plt.show()

plt.figure()
# Plot loss curve
plt.plot(loss_list)
plt.title("RNN Loss Curve", fontsize=18)
plt.xlabel("Iteration", fontsize=18)
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
#############################################################################################
#############################################################################################
#                                                                                           #
#                                        Testing                                            #
#                                                                                           #
#############################################################################################
#############################################################################################
print("Testing... ")
acc = 0
for i in range(TEST_MAX):
    a_dec = np.random.randint(largest/2)
    b_dec = np.random.randint(largest/2)
    c_dec = a_dec + b_dec # c = a + b
    # In binary
    a_bin = binary[a_dec]
    b_bin = binary[b_dec]
    c_bin = binary[c_dec]
    if forward(a_bin, b_bin) == c_dec:
        acc+=1

print(acc/float(TEST_MAX)*100)
