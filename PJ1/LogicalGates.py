import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

x= np.array([0,1,2,3])

j=np.arange(0,6,0.000001)


def AND(x1,x2):
    w1,w2,theta = .5, .5, .7
    tmp = x1*w1+x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
def OR(x1,x2):
    w1,w2,theta = .5, .5, 0
    tmp = x1*w1+x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
def NAND(x1,x2):
    w1,w2,theta = .5, .5, 0.7
    tmp = x1*w1+x2*w2
    if tmp <= theta:
        return 1
    elif tmp > theta:
        return 0
    
def NOR(x1,x2):
    w1,w2,theta = .5, .5, 0
    tmp = x1*w1+x2*w2
    if tmp <= theta:
        return 1
    elif tmp > theta:
        return 0

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

def XNOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = NAND(s1,s2)
    return y



print("And  gate:", AND(0,0), AND(0,1), AND(1,0), AND(1,1))
print("Or   gate:", OR(0,0), OR(0,1), OR(1,0), OR(1,1))
print("Nand gate:", NAND(0,0), NAND(0,1), NAND(1,0), NAND(1,1))
print("Nor  gate:", NOR(0,0), NOR(0,1), NOR(1,0), NOR(1,1))
print("Xor  gate:", XOR(0,0), XOR(0,1), XOR(1,0), XOR(1,1))
print("Xnor gate:", XNOR(0,0), XNOR(0,1), XNOR(1,0), XNOR(1,1))




