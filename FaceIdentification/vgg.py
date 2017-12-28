# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 11:05:03 2017

@author: Insight
"""

import tensorflow as tf

class VGG(object):
    '''
    @description: 构建VGG-Net16网络结构
    @params:
        - weight_decay: 网络权值正则化系数
        - keep_prob: dropout 的概率
        - num_classes: 分类数目
    @return: 返回网络预测的结果
    '''
    def __init__(self,weight_decay,keep_prob,num_classes):
        self._regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        self._keep_prob = keep_prob
        self._num_classes = num_classes
        self._mean_pixel = [103.939, 116.779, 123.68]  #需要根据训练样本进行计算
        
    def vgg16(self,images):
        
        red, green, blue = tf.split(images,num_or_size_splits=3,axis=3)
        
        x = tf.concat([red - self._mean_pixel[0],\
                       green - self._mean_pixel[1],\
                       blue - self._mean_pixel[2]], axis=3)
        
        x = self._conv_layer('conv1_1',x,64)
        x = self._conv_layer('conv1_2',x,64)
        x = self._max_pool('pool1',x)
        
        x = self._conv_layer('conv2_1',x,128)
        x = self._conv_layer('conv2_2',x,128)
        x = self._max_pool('pool2',x)
        
        x = self._conv_layer('conv3_1',x,256)
        x = self._conv_layer('conv3_2',x,256)
        x = self._conv_layer('conv3_3',x,256)   
        x = self._max_pool('pool3',x)
        
        x = self._conv_layer('conv4_1',x,512)
        x = self._conv_layer('conv4_2',x,512)
        x = self._conv_layer('conv4_3',x,512)
        x = self._max_pool('pool4',x)
        
        x = self._conv_layer('conv5_1',x,512)
        x = self._conv_layer('conv5_2',x,512)
        x = self._conv_layer('conv5_3',x,512)
        x = self._max_pool('pool5',x)
            
        #稍作修改，输入图片是50*50，堆叠卷积输出是2*2*512，全连接层输出设置为1024
        x = tf.nn.relu(self._fully_connected('fc6',x,1024))
        with tf.name_scope('dropout1'):    
            x = tf.nn.dropout(x,self._keep_prob)
        
        x = tf.nn.relu(self._fully_connected('fc7',x,1024))
        with tf.name_scope('dropout2'):
            x = tf.nn.dropout(x,self._keep_prob)
        
        logits = self._fully_connected('fc8',x,self._num_classes)
        
        prob = tf.nn.softmax(logits,name='softmax')
        
        return logits, prob
    
    def _conv_layer(self,layer_name,x,out_channels,ksize=[3,3],stride=[1,1,1,1],trainable=True):
        in_channels = x.get_shape().as_list()[-1]
        with tf.variable_scope(layer_name):
            weights = tf.get_variable(name='weights',
                                      shape=[ksize[0],ksize[1],in_channels,out_channels],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      regularizer=self._regularizer,
                                      trainable=trainable)
            biases = tf.get_variable(name='biases',
                                     shape=[out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            y = tf.nn.relu(tf.nn.conv2d(x,weights,stride,padding='SAME')+biases)
            return y
    
    def _max_pool(self,layer_name,x,ksize=[1,2,2,1],stride=[1,2,2,1]):
        with tf.name_scope(layer_name):
            y = tf.nn.max_pool(x,ksize,stride,padding='SAME',name='pool')
            return y
    
    def _batch_norm(self,layer_name,x):
        with tf.name_scope(layer_name):
            epsilon = 1e-3
            batch_mean, batch_var = tf.nn.moments(x,[0])
            y = tf.nn.batch_normalization(x,
                                          mean=batch_mean,
                                          variance=batch_var,
                                          offset=None,
                                          scale=None,
                                          variance_epsilon=epsilon)
            return y
    
    def _fully_connected(self,layer_name,x,out_size):
        with tf.variable_scope(layer_name):
            shape = x.get_shape().as_list()
            in_size = 1
            for d in shape[1:]:
                in_size *= d
            temp = tf.reshape(x,[-1,in_size])
            weights = tf.get_variable(name='weights',
                                      shape=[in_size,out_size],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      regularizer=self._regularizer)
            biases = tf.get_variable(name='biases',
                                     shape=[out_size],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            y = tf.nn.bias_add(tf.matmul(temp,weights),biases)
            return y
        
        