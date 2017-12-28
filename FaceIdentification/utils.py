# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:29:04 2017

@author: Insight
"""

import tensorflow as tf

def calc_loss(logits,labels):
    '''
    @description: 计算loss
    @params:
        - logits: 样本的预测值
        - labels: 样本的真实标记
    @return: 返回计算的loss
    '''
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits,
                                                                       name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = cross_entropy_mean + regularization_loss
    return loss

def calc_accuracy(logits,labels):
    '''
    @description: 计算模型预测准确率
    @params:
        - logits: 样本的预测值
        - labels: 样本的真实标记
    @return: 返回计算的accuracy
    '''
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits,1),labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float16)) * 100
    return accuracy

def optimizer(lr_base,step_size,lr_decay,loss):
    '''
    @description: 定义网络优化算法
    @params:
        - lr_base: 初始学习率
        - step_size: 学习率衰减速度（每迭代step_size步，学习率衰减一次）
        - lr_decay: 学习率衰减系数    lr = lr_base * lr_decay ^ (global_step / step_size)
    @return: 返回优化操作，学习率，global_step
    '''
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0,name='global_step',trainable=False)
        learning_rate = tf.train.exponential_decay(
                                               lr_base,
                                               global_step,
                                               step_size,
                                               lr_decay,
                                               staircase=True
                                               )
        optimizer = tf.train.MomentumOptimizer(learning_rate,0.9)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op,learning_rate,global_step
