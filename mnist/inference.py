# -*- coding: utf-8 -*-

import numpy as np
import os
import pickle
from PIL import Image
import time


def inference(images_path):
    '''
    @description: 对样本进行预测
    @params: 
    - images_path: 样本图片所在路径
    @return: 
    - y_predict: 预测的样本的标记
    '''
    #加载已训练模型
    model = './model/model.pkl'
    
    #加载数据
    files = os.listdir(images_path)
    num_examples = len(files)
    with open(model,'rb') as file:
        lenet = pickle.load(file)
        
    y_predict = np.zeros(num_examples,dtype=np.int)
    start_time = time.time()
    for i in range(num_examples):
        file_name = images_path + files[i]
        try:
            image = Image.open(file_name)
            image = np.reshape(image,(-1,28,28,1))
        except Exception as e:
            print(e)
            break
        else:
            image = np.array(image,dtype=np.float32) / 255. #归一化
            #预测
            y_predict[i] = lenet.predict(image)     
    duration = time.time() - start_time
    mean_time = duration / num_examples
    print('It took %.3f seconds, each picture took %.3f seconds'%(duration,mean_time))
    return y_predict

if __name__ == '__main__':
    images_path = './data/handwriting_digits_images/'
    y_predict = inference(images_path)
    
    
    