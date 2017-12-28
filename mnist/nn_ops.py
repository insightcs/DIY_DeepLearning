# -*- coding: utf-8 -*-

import numpy as np

class NN(object):
    
    def relu(data):
        '''
        @description: relu（rectified linear units）函数的实现
        @params: 
        - data:
        @return: 
        '''
        return np.maximum(data,0)
    
    def softmax(data):
        '''
        @description: softmax函数的实现
        @params: 
        - data:
        @return: 
        '''
        temp = np.sum(np.exp(data),axis=1)
        data = np.divide(np.exp(data),np.repeat(temp,data.shape[1]).reshape(data.shape[0],\
                         data.shape[1]))
        return data
    
    def maxpool(x,ksize,stride,pad=0):
        '''
        @description: max-pooling的简单实现
        @params: 
        - x: 输入的数据(N, H, W, C)
        - ksize: pooling 的窗大小(ksize,ksize)
        - stride: window每次移动的步长
        - pad：为输入数据zero-padding的数量,默认为0
        @return: 
        - out: max-pooling后的数据，out.shape=(N, Ho, Wo, F)
        Ho = 1 + (H + 2 * pad - ksize) / stride
        Wo = 1 + (W + 2 * pad - ksize) / stride
        '''
        out = None
        N,H,W,C = x.shape
        Ho = int(1 + (H + 2 * pad - ksize) / stride)
        Wo = int(1 + (W + 2 * pad - ksize) / stride)
        x_pad = np.zeros((N,H+2*pad,W+2*pad,C))
        x_pad[:,pad:pad+H,pad:pad+W,:] = x
        out = np.zeros((N,Ho,Wo,C))
        for i in range(Ho):
            for j in range(Wo):
                for c in range(C):
                    out[:,i,j,c] = np.max(x_pad[:,i*stride:i*stride+ksize,\
                       j*stride:j*stride+ksize,c], axis=(1, 2))
        return out
    
    def conv2d(x, w, stride,pad=0):
        '''
        @description:2d卷积神经网络的实现
        @params:
        - x: 输入数据 (N, H, W, C)
        - w: 卷积核参数 (HH, WW, C, F)
        - stride: 卷积核每次移动的步长
        - pad: 为输入数据zero-padding的数量,默认为0
        @returns:
        - out: 卷积之后的数据,out.shape=(N, Ho, Wo, F)
        Ho = 1 + (H + 2 * pad - HH) / stride
        Wo = 1 + (W + 2 * pad - WW) / stride
        '''
        out = None
        assert x.shape[3] == w.shape[2],('x.shape:%s,w.shape:%s'%(x.shape,w.shape))
        N,H,W,C = x.shape
        HH,WW,C,F = w.shape
        Ho = int(1 + (H + 2 * pad - HH) / stride)
        Wo = int(1 + (W + 2 * pad - WW) / stride)
        x_pad = np.zeros((N,H+2*pad,W+2*pad,C))
        x_pad[:,pad:pad+H,pad:pad+W,:] = x
        
        out = np.zeros((N,Ho,Wo,F))
        
        for i in range(Ho):
            for j in range(Wo):
                for f in range(F):
                    # N*HH*WW*C, HH*WW*C = N*HH*WW*C, sum -> N*1
                    out[:,i,j,f] = np.sum(x_pad[:,i*stride:i*stride+HH,\
                       j*stride:j*stride+WW,:] * w[:, :, :,f], axis=(1, 2, 3)) 
        return out
        
    def cross_entropy(predicts,labels):
        '''
        @description: 计算交叉熵
        @params: 
        - predicts: 预测的样本的概率分布
        - labels: 样本的真是标记
        @return: 
        - cross_entropy: 计算的交叉熵的结果
        '''
        cross_entropy = -np.sum(labels * np.log(predicts))
        return cross_entropy
        
        
        
        