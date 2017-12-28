# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 21:48:35 2017

@author: Insight
"""
import tensorflow as tf


class ResNet(object):
    '''
    @description: 构建ResNet50网络结构
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
    
    def res_net_50(self,images,is_train=True):
        is_train = tf.constant(is_train,dtype=tf.bool)
        red, green, blue = tf.split(images,num_or_size_splits=3,axis=3)
        
        x = tf.concat([red - self._mean_pixel[0],\
                       green - self._mean_pixel[1],\
                       blue - self._mean_pixel[2]], axis=3)
        
        with tf.variable_scope('conv1'):
            x = self._conv_layer('conv1_1',x,64,ksize=[7,7],stride=[1,2,2,1],padding='SAME')
            x = self._batch_norm('conv1_bn1',x,is_train)
            x = tf.nn.relu(x)
            
        x = self._pool('pool1',x,ksize=[1,3,3,1],stride=[1,2,2,1])
            
        with tf.variable_scope('conv2') as scope:
            x = self._projection_block(scope.name,x,[64,64,256],is_train,stride=[1,1,1,1],stage='1')
            x = self._identity_block(scope.name,x,[64,64,256],is_train,stage='2')
            x = self._identity_block(scope.name,x,[64,64,256],is_train,stage='3')
            
        with tf.variable_scope('conv3') as scope:
            x = self._projection_block(scope.name,x,[128,128,512],is_train,stage='1')
            x = self._identity_block(scope.name,x,[128,128,512],is_train,stage='2')
            x = self._identity_block(scope.name,x,[128,128,512],is_train,stage='3')
            x = self._identity_block(scope.name,x,[128,128,512],is_train,stage='4')
            
        with tf.variable_scope('conv4') as scope:
            x = self._projection_block(scope.name,x,[256,256,1024],is_train,stage='1')
            x = self._identity_block(scope.name,x,[256,256,1024],is_train,stage='2')
            x = self._identity_block(scope.name,x,[256,256,1024],is_train,stage='3')
            x = self._identity_block(scope.name,x,[256,256,1024],is_train,stage='4')
            x = self._identity_block(scope.name,x,[256,256,1024],is_train,stage='5')
            x = self._identity_block(scope.name,x,[256,256,1024],is_train,stage='6')
        
        with tf.variable_scope('conv5') as scope:
            x = self._projection_block(scope.name,x,[512,512,2048],is_train,stage='1')
            x = self._identity_block(scope.name,x,[512,512,2048],is_train,stage='2')
            x = self._identity_block(scope.name,x,[512,512,2048],is_train,stage='3')
            
        #稍作修改，输入图片是50*50，堆叠卷积输出是2*2*2048，所以mean-pooling的ksize设置为2
        x = self._pool('pool2',x,ksize=[1,2,2,1],stride=[1,1,1,1],is_max_pool=False)

        logits = self._fully_connected('fc6',x,self._num_classes)
        
        prob = tf.nn.softmax(logits,name='softmax')
        
        return logits, prob
    
    def _identity_block(self,block_name,x,out_channels,is_train,stage=None):
        
        base_name = block_name + '_' + stage
        
        y = self._conv_layer(base_name+'_1',x,out_channels[0],ksize=[1,1],\
                             stride=[1,1,1,1])
        y = self._batch_norm(base_name+'_bn1',y,is_train)
        y = tf.nn.relu(y,name='relu')
        
        y = self._conv_layer(base_name+'_2',y,out_channels[1],ksize=[3,3],\
                             stride=[1,1,1,1],padding='SAME')
        y = self._batch_norm(base_name+'_bn2',y,is_train)
        y = tf.nn.relu(y,name='relu')
        
        y = self._conv_layer(base_name+'_3',y,out_channels[2],ksize=[1,1],\
                             stride=[1,1,1,1])
        y = self._batch_norm(base_name+'_bn3',y,is_train)
        
        y = tf.add(x,y)
        y = tf.nn.relu(y,name='relu')
        return y
    
    def _projection_block(self,block_name,x,out_channels,is_train,stride=[1,2,2,1],stage=None):
        
        base_name = block_name + '_' + stage
        
        y = self._conv_layer(base_name+'_1',x,out_channels[0],ksize=[1,1],\
                             stride=stride)
        y = self._batch_norm(base_name+'_bn1',y,is_train)
        y = tf.nn.relu(y,name='relu')
        
        y = self._conv_layer(base_name+'_2',y,out_channels[1],ksize=[3,3],\
                             stride=[1,1,1,1],padding='SAME')
        y = self._batch_norm(base_name+'_bn2',y,is_train)
        y = tf.nn.relu(y,name='relu')
        
        y = self._conv_layer(base_name+'_3',y,out_channels[2],ksize=[1,1],\
                             stride=[1,1,1,1])
        y = self._batch_norm(base_name+'_bn3',y,is_train)
        
        shortcut = self._conv_layer(base_name+'_1_',x,out_channels[2],ksize=[1,1],\
                             stride=stride)
        shortcut = self._batch_norm(base_name+'_bn1_',shortcut,is_train)
        
        y = tf.add(y,shortcut)
        y = tf.nn.relu(y,name='relu')
        return y
   
    def _conv_layer(self,layer_name,x,out_channels,ksize,stride,padding='VALID',trainable=True):
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
            y = tf.nn.conv2d(x,weights,stride,padding) + biases
            return y
    
    def _pool(self,layer_name,x,ksize,stride,padding='VALID',is_max_pool=True):
        with tf.name_scope(layer_name):
            if is_max_pool:
                y = tf.nn.max_pool(x,ksize,stride,padding,name='max_pool')
            else:
                y = tf.nn.avg_pool(x,ksize,stride,padding,name='mean_pool')
            return y
    
    def _batch_norm(self,layer_name,x,is_train,convolution=True):
        with tf.variable_scope(layer_name):
            epsilon = 1e-3
            offset = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]),name='beta')
            scale = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]),name='gamma')
            if convolution:
                batch_mean, batch_var = tf.nn.moments(x,[0,1,2],name='moments')
            else:
                batch_mean, batch_var = tf.nn.moments(x,[0],name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
                
            mean,var = tf.cond(is_train,mean_var_with_update,lambda:(ema.average(batch_mean),
                                                                     ema.average(batch_var)))
            y = tf.nn.batch_normalization(x,
                                          mean=mean,
                                          variance=var,
                                          offset=offset,
                                          scale=scale,
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
            y = tf.matmul(temp,weights) + biases
            return y
        