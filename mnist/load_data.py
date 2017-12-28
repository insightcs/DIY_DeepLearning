# -*- coding: utf-8 -*-

import numpy as np
import gzip

class DataSet(object):
    '''
    @description: 将数据集定义为类，方便读取，包含数据归一化操作
    '''
    def __init__(self,image,label):
        assert image.shape[0] == label.shape[0],(
        'image.shape:%s label.shape:%s'%(image.shape,label.shape))
        self._num_examples = image.shape[0]
        self._image = image / 255.
        self._label = label
        self._index_in_epoch = 0
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def image(self):
        return self._image
    
    @property
    def label(self):
        return self._label
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start==0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._image = self._image[perm]
            self._label = self._label[perm]
        if start+batch_size>self._num_examples:
            rest_num_examples = self._num_examples - start
            image_rest_part = self._image[start:self._num_examples]
            label_rest_part = self._label[start:self._num_examples]
        
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._image = self._image[perm]
            self._label = self._label[perm]
        
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            image_new_part = self._image[start:end]
            label_new_part = self._label[start:end]
            return np.concatenate((image_rest_part,image_new_part),axis=0),\
        np.concatenate((label_rest_part,label_new_part),axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._image[start:end],self._label[start:end]

def read4bytes(bytestream):
    '''
    @description: 从字节流中读取4个字节数据
    @params: 
    - bytestream：字节流
    @return: 返回读取的数据
    '''
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def convert_to_one_hot(labels, num_classes):
    '''
    @description: 将样本的labels转化为one-hot编码.
    @params: 
    - labels：样本的标记
    - num_classes：样本的类别数
    @return: 返回one-hot编码后的标记
    '''
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

def extract_images(file):
    '''
    @description: 从压缩文件中读取图片数据.
    @params: 
    - file：压缩文件
    @return: 返回读取的图片数据
    '''
    print('Extracting', file.name)
    with gzip.GzipFile(fileobj=file) as bytestream:
        magic = read4bytes(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % \
                         (magic, file.name))
        num_images = read4bytes(bytestream)
        rows = read4bytes(bytestream)
        cols = read4bytes(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

def extract_labels(file,one_hot=False,num_classes=10):
    '''
    @description: 从压缩文件中读取样本标记信息.
    @params: 
    - file：压缩文件
    - one-hot: 是否将标记转化为one-hot编码，默认不转化
    - num_classes：样本的类别数
    @return: 返回读取的样本标记
    '''
    print('Extracting', file.name)
    with gzip.GzipFile(fileobj=file) as bytestream:
        magic = read4bytes(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, file.name))
        num_items = read4bytes(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return convert_to_one_hot(labels, num_classes)
        return labels

def LoadDataSet(file_path):
    '''
    @description: 解压数据集.gz文件，加载数据集
    @params: file_path---压缩文件的路径
    @return: train_dataset---训练集类
             valid_dataset---验证集类
             test_dataset---测试集类
    '''
    train_labels_file = file_path + 'train-labels-idx1-ubyte.gz'
    train_images_file = file_path + 'train-images-idx3-ubyte.gz'
    test_labels_file = file_path + 't10k-labels-idx1-ubyte.gz'
    test_images_file = file_path + 't10k-images-idx3-ubyte.gz'
    with open(train_labels_file,'rb') as file:
        train_labels = extract_labels(file,one_hot=True)
    with open(train_images_file,'rb') as file:
        train_images = extract_images(file)
    with open(test_labels_file,'rb') as file:
        test_labels = extract_labels(file,one_hot=True)
    with open(test_images_file,'rb') as file:
        test_images = extract_images(file)

    valid_labels = train_labels[0:500]
    valid_images = train_images[0:500]
    train_labels = train_labels[500:]
    train_images = train_images[500:]

    train_dataset = DataSet(train_images,train_labels)
    valid_dataset = DataSet(valid_images,valid_labels)
    test_dataset = DataSet(test_images,test_labels)
    return train_dataset, valid_dataset, test_dataset

def LoadTestDataSet(file_path):
    '''
    @description: 解压数据集.gz文件，加载测试数据集
    @params: file_path---压缩文件的路径
    @return: test_dataset---测试集类
    '''
    test_labels_file = file_path + 't10k-labels-idx1-ubyte.gz'
    test_images_file = file_path + 't10k-images-idx3-ubyte.gz'
    with open(test_labels_file,'rb') as file:
        test_labels = extract_labels(file,one_hot=False)
    with open(test_images_file,'rb') as file:
        test_images = extract_images(file)

    test_dataset = DataSet(test_images,test_labels)
    return test_dataset
