# -*- coding: utf-8 -*-

import numpy as np
import pickle
from load_data import LoadTestDataSet

def test():
    '''
    @description: 测试网络模型，并输出测试错误率
    @params:
    @return: 
    '''
    #加载模型
    model = './model.pkl'
    
    #加载数据集
    test_dataset = LoadTestDataSet('./data/')
    with open(model,'rb') as file:
        lenet = pickle.load(file)
    #预测
    y_predict = lenet.predict(test_dataset.image)
    #计算准确率
    num_correct = np.equal(y_predict,test_dataset.label).sum()
    accuracy = num_correct / test_dataset.num_examples
    print('testing accuracy: %.2f%%'%(accuracy*100))
    print('testing error: %.2f%%'%((1-accuracy)*100))

if __name__ == '__main__':
    test()
        