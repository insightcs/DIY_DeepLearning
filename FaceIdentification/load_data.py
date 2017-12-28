# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:15:18 2017

@author: Insight
"""

import tensorflow as tf
import os
from PIL import Image
import numpy as np
import random

def split_datasets(file_dir,saved_dir):
    '''
    @description:划分数据集，将划分的图片对应的path name和label保存到txt文件中
    @params:
        - file_dir:人脸图片数据集所在的路径
        - saved_dir:划分的结果要保存的路径
    @return: None
    '''
    train_files = []
    test_files = []
    label = 0
    folders = os.listdir(file_dir)
    for folder in folders:
        files = os.listdir(file_dir+folder)
        num = len(files)
        if num < 40:#每个人的图片小于40张就舍弃
            continue
        temp = []
        for file in files:
            file_name = file_dir + folder + '/' + file
            image = Image.open(file_name)
            if image.size[0] < 50 or image.size[1] < 50:#图像尺寸小于50*50的舍弃掉
                continue
            temp.append([file_name,label])
        if len(temp) < 5:
            continue
        num_train = int(np.floor(len(temp)*0.95))
        random.shuffle(temp)
        train_files.extend(temp[0:num_train]) 
        test_files.extend(temp[num_train:])
        label += 1
        
    random.shuffle(train_files)    
    random.shuffle(test_files)
    valid_files = test_files[0:1000]
    test_files = test_files[1000:]
    print('The number of people is %s'%label)
    np.savetxt(saved_dir+'train_list.txt',np.array(train_files),fmt='%s')
    np.savetxt(saved_dir+'valid_list.txt',np.array(valid_files),fmt='%s')
    np.savetxt(saved_dir+'test_list.txt',np.array(test_files),fmt='%s')

def get_batch(file_name,image_W,image_H,batch_size,is_train=True):
    '''
    @description:从数据集中读取batch_size大小的数据
    @params:
        - file_name:数据集对应的txt文件名
        - image_W: 输入网络的图片的宽度
        - image_H: 输入网络的图片的高度
        - batch_size: 每次要读取的数据的大小
        - is_train: 要读取的数据是否是训练数据（训练数据需要做数据增强操作）
    @return: 返回读取的数据(image_batch,label_batch)
    '''
    
    train_files = np.loadtxt(file_name,dtype=np.str)
    image = train_files[:,0]
    label = train_files[:,1].astype(int)
    
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int64)

    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    
    #读取图片
#    image = tf.image.decode_image(image_contents,channels=3)
    image = tf.image.decode_jpeg(image_contents,channels=3)
    
    #图片裁剪
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image,[image_W,image_H])
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    
    if is_train:
        #图像增强操作
        image = tf.image.random_flip_left_right(image)#随机水平翻转
        image = tf.image.random_brightness(image, max_delta=0.2)#随机调整亮度
#        image = tf.image.random_hue(image, max_delta=0.05)#随机调整色调
#        image = tf.image.random_contrast(image, lower=0.2, upper=0.5)#随机调整对比度
#        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)#随机调整饱和度
        
        min_after_dequeue = 500
        capacity = min_after_dequeue + 3 * batch_size
        image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                         batch_size=batch_size,
                                                         capacity=capacity,
                                                         min_after_dequeue=min_after_dequeue,
                                                         num_threads=16)
    else:
        image_batch,label_batch = tf.train.batch([image,label],
                                                 batch_size=batch_size,
                                                 num_threads=16,
                                                 capacity=batch_size)
        
    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch
    

if __name__ == '__main__':
    split_datasets(file_dir='../webface/',saved_dir='./data/')
        
        
     
    