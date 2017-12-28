# -*- coding: utf-8 -*-

import numpy as np
from nn_ops import NN as nn


class LeNet(object):
    '''
    @description: 构建LeNet网络模型，实现网络前向传播，后向传播，参数更新，、
    计算损失loss，计算准确率
    '''
    def __init__(self,input_params,conv_params,pool_params,full_params,\
                 output_params,optimizer='normal'):
        '''
        @description: 网络参数初始化
        @params: 
        - input_params: 网络输入层的参数
        - conv_params: 卷积层参数，包括卷积核大小，步长，pad，卷积核数量
        - pool_params: pooling层参数，包括pool window大小，步长，pad
        - full_params: 全连接层的参数
        - output_params：输出层的参数，包括分类数
        - optimizer: 网络参数优化方法,'normal'对应SGD优化方法，\
                     'Momentum'对应SGD+动量项（momentum），默认为'normal'
        @return: 
        '''
        self._input_height = input_params[0]
        self._input_width = input_params[1]
        self._input_channels = input_params[2]
        self._conv_ksize = conv_params['ksize']
        self._conv_stride = conv_params['stride']
        self._conv_pad = conv_params['pad']
        self._conv_num = conv_params['num']
        self._pool_ksize = pool_params['ksize']
        self._pool_stride = pool_params['stride']
        self._pool_pad = pool_params['pad']
        
        self._full_params = full_params
        self._output_params = output_params
        
        self._weights = {}
        self._biases = {}
        self._weights[1] = np.random.normal(0.0,0.01,(self._conv_ksize[0],\
                     self._conv_ksize[1],self._input_channels,self._conv_num))
        self._weights[4] = np.random.normal(0.0,0.01,(self._full_params[0],self._full_params[1]))
        self._weights[5] = np.random.normal(0.0,0.01,(self._output_params[0],self._output_params[1]))
        self._biases[1] = np.zeros(self._conv_num)
        self._biases[4] = np.zeros(self._full_params[1])
        self._biases[5] = np.zeros(self._output_params[1])
        
        self._grad_weights = {}
        self._grad_biases = {}
        self._grad_weights[1] = np.zeros_like(self._weights[1])
        self._grad_weights[4] = np.zeros_like(self._weights[4])
        self._grad_weights[5] = np.zeros_like(self._weights[5])
        self._grad_biases[1] = np.zeros_like(self._biases[1])
        self._grad_biases[4] = np.zeros_like(self._biases[4])
        self._grad_biases[5] = np.zeros_like(self._biases[5])
        
        self._optimizer = optimizer
        
        if self._optimizer == 'Momentum':
            self._v_weights = {}
            self._v_biases = {}
            self._v_weights[1] = np.zeros_like(self._weights[1])
            self._v_weights[4] = np.zeros_like(self._weights[4])
            self._v_weights[5] = np.zeros_like(self._weights[5])
            self._v_biases[1] = np.zeros_like(self._biases[1])
            self._v_biases[4] = np.zeros_like(self._biases[4])
            self._v_biases[5] = np.zeros_like(self._biases[5])
        
        self._x = {}
        self._delta = {}
    
    def forward(self,x):
        
        self._x[1] = x
        
        self._x[2] = nn.relu(nn.conv2d(self._x[1],self._weights[1],\
               stride=self._conv_stride,pad=self._conv_pad) + self._biases[1])
        
        self._x[3] = nn.maxpool(self._x[2],ksize=self._pool_ksize,\
               stride=self._pool_stride,pad=self._pool_pad)
        
        self._x[4] = np.reshape(self._x[3],(-1,self._full_params[0]))
        
        self._x[5] = nn.relu(np.dot(self._x[4],self._weights[4]) + self._biases[4])
        
        self._x[6] = nn.softmax(np.dot(self._x[5],self._weights[5]) + self._biases[5])
        
        return self._x[6]
        
    def backforward(self,labels):
        m = labels.shape[0]
        
        self._delta[6] = self._x[6] - labels
        self._grad_weights[5] = np.dot(self._x[5].T,self._delta[6]) / m
        self._grad_biases[5] = np.sum(self._delta[6],axis=0) / m
        
        self._delta[5] = np.dot(self._delta[6],self._weights[5].T) * (self._x[5] > 0)
        self._grad_weights[4] = np.dot(self._x[4].T,self._delta[5]) / m
        self._grad_biases[4] = np.sum(self._delta[5],axis=0) / m
        
        self._delta[4] = np.dot(self._delta[5],self._weights[4].T)
        
        self._delta[4] = self._delta[4].reshape(self._x[3].shape[0],self._x[3].shape[1],\
                   self._x[3].shape[2],self._x[3].shape[3])
        
        stride = self._pool_stride
        N,H,W,C = self._x[3].shape
        self._delta[2] = np.zeros(self._x[2].shape)
                      
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        array = self._x[2][n,h*stride:h*stride+self._pool_ksize,\
                                       w*stride:w*stride+self._pool_ksize,c]
                        max_i = 0
                        max_j = 0
                        max_value = array[0,0]
                        for i in range(self._pool_ksize):
                            for j in range(self._pool_ksize):
                                if array[i,j]>max_value:
                                    max_value = array[i,j]
                                    max_i = i
                                    max_j = j
                        self._delta[2][n,h*stride+max_i,w*stride+max_j,c] \
                                    = self._delta[4][n,h,w,c]
        self._delta[2] = self._delta[2] * (self._x[2] > 0)
        
        stride = self._conv_stride
        pad = self._conv_pad
        N,H,W,C = self._x[1].shape
        _,Ho,Wo,_ = self._delta[2].shape
        HH,WW,_,F = self._weights[1].shape
        
        H1 = H - HH + 2 * pad + 1
        W1 = W - WW + 2 * pad + 1
        delta_pad = np.zeros((N,H1,W1,F))
        for i in range(Ho):
            for j in range(Wo):
                delta_pad[:,i*stride,j*stride,:] = self._delta[2][:,i,j,:]
        
        x_pad = np.zeros((N,H+2*pad,W+2*pad,C))
        x_pad[:,pad:pad+H,pad:pad+W,:] = self._x[1]
        
        for f in range(F):
            for c in range(C):
                for h in range(HH):
                    for w in range(WW):
                        self._grad_weights[1][h,w,c,f] = np.sum(x_pad[:,h:h+H1,w:w+W1,c] *\
                        delta_pad[:,:,:,f],axis=(0,1,2))
        
        self._grad_weights[1] = self._grad_weights[1] / m
        self._grad_biases[1] = np.sum(self._delta[2],axis=(0,1,2)) / m
    
    def optimizer(self,labels,learning_rate):
        
        self.backforward(labels)
        # update
        if self._optimizer == 'normal':
            self._weights[1] = self._weights[1] - learning_rate * self._grad_weights[1]
            self._biases[1] = self._biases[1] - learning_rate * self._grad_biases[1]
            
            self._weights[4] = self._weights[4] - learning_rate * self._grad_weights[4]
            self._biases[4] = self._biases[4] - learning_rate * self._grad_biases[4]
            
            self._weights[5] = self._weights[5] - learning_rate * self._grad_weights[5]
            self._biases[5] =  self._biases[5] - learning_rate * self._grad_biases[5]
        elif self._optimizer == 'Momentum':
            self._v_weights[1] = 0.1 * self._v_weights[1] - learning_rate * self._grad_weights[1]
            self._v_biases[1] = 0.1 * self._v_biases[1] - learning_rate * self._grad_biases[1]
            self._v_weights[4] = 0.1 * self._v_weights[4] - learning_rate * self._grad_weights[4]
            self._v_biases[4] = 0.1 * self._v_biases[4] - learning_rate * self._grad_biases[4]
            self._v_weights[5] = 0.1 * self._v_weights[5] - learning_rate * self._grad_weights[5]
            self._v_biases[5] = 0.1 * self._v_biases[5] - learning_rate * self._grad_biases[5]
            
            self._weights[1] = self._weights[1] + self._v_weights[1]
            self._biases[1] = self._biases[1] + self._v_biases[1]
            self._weights[4] = self._weights[4] + self._v_weights[4]
            self._biases[4] = self._biases[4] + self._v_biases[4]
            self._weights[5] = self._weights[5] + self._v_weights[5]
            self._biases[5] = self._biases[5] + self._v_biases[5]
        else:
            raise ValueError('The %s optimization method is non-existenct !'%self._optimizer)
    
    def predict(self,x):
        x = np.reshape(x,(-1,self._input_height,self._input_width,self._input_channels))
        logits = self.forward(x)
        y_predict = np.argmax(logits,axis=1)
        return y_predict
    
    def calc_loss(self,logits,labels):
        loss = nn.cross_entropy(logits,labels)
        return loss
        
    def calc_accuracy(self,logits,labels):
        y_predict = np.argmax(logits,axis=1)
        y = np.argmax(labels,axis=1)
        num_correct = np.equal(y_predict,y).sum()
        accuracy = num_correct / labels.shape[0]
        return accuracy
    
    