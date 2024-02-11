import numpy as np

def softmax_origin(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def softmax(a):
    a = a - np.max(a)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([1010, 1000, 990])

print(softmax(a))
print(np.sum(softmax(a)))

#result:
#[9.99954600e-01 4.53978686e-05 2.06106005e-09]
#1.0

#softmax結果可以視為各項機率之統計
#皆以a之最大為輸出結果，因此學習階段完成後，在推論階段會選擇跳過softmax步驟