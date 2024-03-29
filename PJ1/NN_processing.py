import numpy as np
import sys, os, pickle
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

def sigmoid(s):
    return 1/(1+np.exp(-s))

def softmax(a):
    a = a - np.max(a)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_test, t_test

def networkSetup():
    with open("sample_weight.pkl", 'rb')as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

x, t = get_data()
network = networkSetup()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("accruary =" + str(float(accuracy_cnt) / len(x)))


x, _ = get_data()
network = networkSetup()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)

#batchingqu 批次處理
x, t = get_data()
network = networkSetup()

batchSize = 100
accuracy_cnt = 0

for i in range(0, len(x), batchSize):
    x_batch = x[i:i+batchSize]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i:i+batchSize])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

