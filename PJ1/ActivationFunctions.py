import numpy as np
import matplotlib.pyplot as plt

#三種激勵函數實作

def step(x):#階梯函數
    y=x>0
    return y.astype(np.float16)

def sigmoid(s):
    return 1/(1+np.exp(-s))

def ReLU(x):
    return np.maximum(0,x)


"""
print("step function")
a=int(input("input: "))
print("output:",step(a))


print("sigmoid function")
a=int(input("input: "))
print("output:",step(a))

print("ReLU function")
a=int(input("input: "))
print("output:",step(a))
"""

x=np.arange(-8,8,0.1)
stepy=step(x)

sigy=sigmoid(x)

ReLUy=ReLU(x)


plt.plot(x,stepy,label="step",color="purple")
plt.plot(x,sigy,label="sigmoid")
plt.plot(x,ReLUy,label="ReLU")

plt.legend()
plt.ylim(-0.1,1.2)
plt.show()