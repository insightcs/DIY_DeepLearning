# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 11:51:59 2017

@author: Insight
"""

import tensorflow as tf
import time
import numpy as np
from load_data import get_batch
from vgg import VGG
import utils

class DCNN(object):
    def __init__(self):
        self._image_H = 50
        self._image_W = 50
        self._image_channels = 3
    
    def train(self,dataset_path,num_classes,batch_size,lr_base,lr_decay,step_size,\
              max_iteration,pretrained_model=None):
        '''
        @description: 构建VGG-Net16网络结构，训练网络模型，输出训练过程中的logs，保存网络模型
        @params:
            - dataset_path: 训练样本集和验证样本集对应的txt文件所在的路径
            - num_classes: 分类数目
            - batch_size: 训练过程中的每次输入网络中的样本数
            - lr_base: 初始学习率
            - lr_decay: 学习率衰减系数
            - step_size: 学习率衰减速度   lr = lr_base * lr_decay ^ (global_step / step_size)
            - max_iteration: 迭代的最大次数
            - pretrained_model: 预训练的模型所在的路径
        @return: None
        '''
        
        train_file_name = dataset_path + 'train_list.txt'
        valid_file_name = dataset_path + 'valid_list.txt'
        
        log_dir = './log/vgg'
        model_dir = './model/vgg'
        
        vgg = VGG(weight_decay=0.0005,keep_prob=0.5,num_classes=num_classes)
        
        train_summary_list = []
        valid_summary_list = []
        
        with tf.Graph().as_default(), tf.device('/gpu:0'):
            
            with tf.name_scope('input'):
                #队列读取训练数据
                train_image,train_label = get_batch(train_file_name,self._image_H,\
                                                    self._image_W,batch_size)
                valid_image,valid_label = get_batch(valid_file_name,self._image_H,\
                                                    self._image_W,250,is_train=False)
    
                x = tf.placeholder(tf.float32,[None,self._image_H,self._image_W,\
                                               self._image_channels],name='x')
                y = tf.placeholder(tf.int64,[None],name='y')
             
            #loss, accuracy, train_op
            logits,_ = vgg.vgg16(x)
            loss = utils.calc_loss(logits,y)
            accuracy = utils.calc_accuracy(logits,y)
            train_op,learning_rate,global_step = utils.optimizer(lr_base,step_size,lr_decay,loss)
            
            #summary
            train_summary_list.append(tf.summary.scalar('train_loss',loss))
            valid_summary_list.append(tf.summary.scalar('valid_loss',loss))
            train_summary_list.append(tf.summary.scalar('train_accuracy',accuracy))
            valid_summary_list.append(tf.summary.scalar('test_accuracy',accuracy))
            train_summary_list.append(tf.summary.scalar('learning rate',learning_rate))
            valid_summary_list.append(tf.summary.scalar('learning rate',learning_rate))
            for var in tf.trainable_variables():
                valid_summary_list.append(tf.summary.histogram(var.name,var))
            train_summary = tf.summary.merge(train_summary_list)
            valid_summary = tf.summary.merge(valid_summary_list)
            
            #session
            saver = tf.train.Saver(max_to_keep=50)
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,\
                                                  log_device_placement=True)) as sess:
                train_writer = tf.summary.FileWriter(log_dir+'train',sess.graph)
                test_writer = tf.summary.FileWriter(log_dir+'valid')
                tf.global_variables_initializer().run()
                tf.local_variables_initializer().run()
                
                #启动多线程
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,coord=coord)
                
                #加载预训练的模型
                if pretrained_model!=None:
                    ckpt = tf.train.get_checkpoint_state(pretrained_model)
                    print('Restoring pretrained model: %s'%ckpt.model_checkpoint_path)
                    saver.restore(sess,ckpt.model_checkpoint_path)
                
                train_time = 0
                for step in range(max_iteration):
                    
                    #模型持久化操作
#                    graph_def = tf.get_default_graph().as_graph_def()
#                    output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['input/x','deepid/Relu'])
#                    with tf.gfile.GFile(model_dir+'deepid_model.pb','wb') as file:
#                        file.write(output_graph_def.SerializeToString())
#                    break
                
                    start_time = time.time()
                    image,label = sess.run([train_image,train_label])
                    _,train_loss,summary_str,train_step = sess.run([train_op,
                                                                    loss,
                                                                    train_summary,
                                                                    global_step],
                                                                    feed_dict={
                                                                            x:image,
                                                                            y:label})
                    train_writer.add_summary(summary_str,global_step=train_step)
                    train_writer.flush()
                    duration = time.time() - start_time
                    train_time += duration
                    
                    #valid and save model
                    if step%1000==0 or (step+1)==max_iteration:
                        image,label = sess.run([valid_image,valid_label])
                        lr,summary_str,valid_loss,validation_accuracy,\
                        train_step = sess.run([learning_rate,
                                               valid_summary,
                                               loss,
                                               accuracy,
                                               global_step],
                                               feed_dict={x:image,y:label})
                        test_writer.add_summary(summary_str,global_step=train_step)
                        test_writer.flush()
                        print('Step %d: train loss = %.3f, valid loss = %.3f,valid accuracy = %.3f%%, lr = %.6f (%.3f sec)'%\
                              (train_step,train_loss,valid_loss,validation_accuracy,\
                               lr,train_time))
                        saver.save(sess,model_dir+'model.ckpt',global_step=train_step)
                        with open(log_dir+'valid_result.txt','at') as file_writer:
                            file_writer.write('%d\t%.3f%%\t%.5f\t%d\r\n'%(train_step,
                                                                          validation_accuracy,
                                                                          lr,
                                                                          train_time))
                #退出多线程
                coord.request_stop()
                coord.join(threads)
                
    def test(self,test_file_path,num_classes,batch_size,model):
        '''
        @description: 构建VGG-Net16网络结构，加载训练好的网络模型，对测试样本进行分类，输出测试准确率
        @params:
            - test_file_path: 测试样本集对应的txt文件所在的路径
            - num_classes: 分类数目
            - batch_size: 测试过程中的每次输入网络中的样本数
            - model: 已经训练好的模型
        @return: None
        '''
        test_file_name = test_file_path + 'test_list.txt'
        vgg = VGG(weight_decay=0.0,keep_prob=1.0,num_classes=num_classes)
        with tf.Graph().as_default(),tf.device('/gpu:0'):
            
            with tf.name_scope('input'):
            #队列读取训练数据
                num_examples = np.loadtxt(test_file_name,dtype=np.str).shape[0]
                test_image,test_label = get_batch(test_file_name,self._image_H,\
                                                  self._image_W,batch_size,\
                                                  is_train=False)  
                
                x = tf.placeholder(tf.float32,[None,self._image_H,self._image_W,\
                                               self._image_channels],name='x')
                
            _, prob = vgg.vgg16(x)
            correct_top_1 = tf.nn.top_k(prob,k=1)
            correct_top_5 = tf.nn.top_k(prob,k=5)
            
            saver = tf.train.Saver()
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                  log_device_placement=True)) as sess:
                saver.restore(sess,model)
                start_time = time.time()
                num_epoch = int(np.ceil(num_examples / batch_size))
                num_examples = num_epoch * batch_size
                index_top_1 = np.zeros(num_examples)
                index_top_5 = np.zeros((num_examples,5))
                label = np.zeros(num_examples)
                for i in range(num_epoch):
                    image,label[i*batch_size:(i+1)*batch_size] = \
                    sess.run([test_image,test_label])
                    index_top_1[i*batch_size:(i+1)*batch_size],\
                    index_top_5[i*batch_size:(i+1)*batch_size] = \
                    sess.run(correct_top_1[1],correct_top_5,feed_dict={x:image})
                duration = time.time() - start_time
                
                top_1 = 0
                top_5 = 0
                for i in range(num_examples):
                    top_1 += label[i] in index_top_1[i]
                    top_5 += label[i] in index_top_5[i]
                top_1 = top_1 / num_examples
                top_5 = top_5 / num_examples
                
                print('top_1 accuracy: %.3f%%, top_5 accuracy: %.3f%%\t%.3fsec)'%\
                     (top_1,top_5,duration))
        
if __name__ == '__main__':
    dcnn = DCNN()
    dcnn.train(dataset_path='./data/',num_classes=3037,batch_size=50,lr_base=0.01,\
               lr_decay=0.95,step_size=100000,max_iteration=400000,\
               pretrained_model=None)
    
    