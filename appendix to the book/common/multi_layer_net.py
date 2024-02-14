﻿# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 載入父目錄檔案的設定
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    """運用全連接的多層神經網路

    Parameters
    ----------
    input_size : 輸入大小（MNIST是784）
    hidden_size_list : 隱藏層的神經元數量清單（e.g. [100, 100, 100]）
    output_size : 輸出大小（MNIST是10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 設定權重的標準差（e.g. 0.01）
        'relu'或'he'是設定「He預設值」
        'sigmoid'或'xavier'是設定「Xavier預設值」
    weight_decay_lambda : Weight Decay（L2 norm）的強度
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 權重初始化
        self.__init_weight(weight_init_std)

        # 產生各層
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """權重的預設值設定

        Parameters
        ----------
        weight_init_std : 設定權重的標準差（e.g. 0.01）
            'relu'或'he'是設定「He預設值」
            'sigmoid'或'xavier'是設定「Xavier預設值」
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU時建議的預設值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid時建議的預設值

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """計算損失函數

        Parameters
        ----------
        x : 輸入資料
        t : 訓練標籤

        Returns
        -------
        損失函數的值
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """計算梯度（數值微分）

        Parameters
        ----------
        x : 輸入資料
        t : 訓練標籤

        Returns
        -------
        擁有各層梯度的字典變數
            grads['W1']、grads['W2']、...是各層的權重
            grads['b1']、grads['b2']、...是各層的偏權值
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """計算梯度（誤差反向傳播法）

        Parameters
        ----------
        x : 輸入資料
        t : 訓練標籤

        Returns
        -------
        擁有各層梯度的字典變數
            grads['W1']、grads['W2']、...是各層的權重
            grads['b1']、grads['b2']、...是各層的偏權值
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
