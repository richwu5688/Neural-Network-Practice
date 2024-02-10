import numpy as np

def sigmoid(s):
    return 1/(1+np.exp(-s))


#LAYER1
print("[LAYER1]")

X = np.array([1.0, 0.5])
print("X.shape:", X.shape)

W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
print("W.shape:", W.shape)

B1 = np.array([0.1, 0.2, 0.3])
print("B1:",B1)

A1 = np.dot(X, W) + B1
print("A1:", A1)

Z1 = sigmoid(A1)
print("Z1:", Z1)
#LAYER1 END

print("\n")

#LAYER2
print("[LAYER2]")
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
print("W2:", W2.shape)

B2 = np.array([0.1, 0.2])
print("B2:", B2.shape)

A2 = np.dot(Z1, W2) + B2
print("A2:", A2)

Z2 = sigmoid(A2)
print("Z2:", Z2)
#LAYER2 END

print("\n")

#LAYER3
print("[LAYER3]")
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
print("W3:", W3.shape)

B3 = np.array([0.1, 0.2])
print("B3:", B3.shape)

A3 = np.dot(Z2, W3) + B3
print("A3:", A3)
#LAYER3 END

Y = A3
print("\nY:", Y)

