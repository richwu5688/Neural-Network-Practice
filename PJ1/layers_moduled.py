import numpy as np

def sigmoid(s):
    return 1/(1+np.exp(-s))

def networkSetup_3Layer():
    ns = {}
    ns['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    ns['b1'] = np.array([0.1, 0.2, 0.3])
    ns['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    ns['b2'] = np.array([0.1, 0.2])
    ns['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    ns['b3'] = np.array([0.1, 0.2])
    return ns
#line 14 wasn't written caused the error message below.
#TypeError: 'NoneType' object is not subscriptable.

def implementNet(net, x):
    W1, W2, W3 = net['W1'], net['W2'], net['W3']
    b1, b2, b3 = net['b1'], net['b2'], net['b3']
    Z1 = sigmoid(np.dot(x, W1) + b1)
    Z2 = sigmoid(np.dot(Z1, W2) + b2)
    y = np.dot(Z2, W3) + b3
    return y

net1 = networkSetup_3Layer()
x = np.array([1.0, 0.5])
y = implementNet(net1, x)
print(y)

