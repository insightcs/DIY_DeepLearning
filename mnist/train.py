# -*- coding: utf-8 -*-

import numpy as np
from load_data import LoadDataSet
from lenet import LeNet
import pickle
import matplotlib.pyplot as plt
            
def train(data_path,batch_size,learning_rate,max_iteration):
    '''
    @description: 训练LeNet网络模型，保存训练好的网络模型
    @params: 
    - data_path: 数据集所在的文件路径
    - batch_size: 每批次的样本数
    - learning_rate: 网络参数的学习速率
    - max_iteration: 最大迭代次数
    @return: 
    '''
    lenet = LeNet(input_params=(28,28,1),
                  conv_params={'ksize':(4,4),'stride':3,'pad':0,'num':20},
                  pool_params={'ksize':2,'stride':2,'pad':0},
                  full_params=(320,100),
                  output_params=(100,10),
                  optimizer='Momentum')
    
    train_dataset,valid_dataset,test_dataset = LoadDataSet(data_path)
    
    train_accuracy = []
    train_loss = []
    valid_accuracy = []
    valid_loss = []
    max_accuracy = 0.0
    for step in range(max_iteration):
        
        image,labels = train_dataset.next_batch(batch_size)
        #前向传播
        predicts = lenet.forward(image)
        
        #计算loss和accuracy
        train_loss.append(lenet.calc_loss(predicts,labels))
        train_accuracy.append(lenet.calc_accuracy(predicts,labels))
        
        # learning rate指数衰减
        if step % 1000 == 0:
            learning_rate = learning_rate * (0.9**int(step/1000))
            
        #后向传播并更新网络参数
        lenet.optimizer(labels,learning_rate)
        
        
        if step % 100 == 0:
            predicts = lenet.forward(valid_dataset.image)
            loss = lenet.calc_loss(predicts,valid_dataset.label)
            accuracy = lenet.calc_accuracy(predicts,valid_dataset.label)
            valid_loss.append(loss)
            valid_accuracy.append(accuracy)
            print('%d-->train loss: %.4f, valid_loss: %.4f, valid accuracy: %.2f%%'%\
                  (step,train_loss[-1],loss,(accuracy*100)))
            
        if ((step + 1) % 1000 == 0) or (step + 1 == max_iteration):
            predicts = lenet.forward(test_dataset.image)
            accuracy = lenet.calc_accuracy(predicts,test_dataset.label)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                file_name = './model/model-' + str(step) + '.pkl'
                with open(file_name,'wb') as file:
                    pickle.dump(lenet,file)
            print('%d-->test accuracy: %.2f%%'%(step,(accuracy*100)))
            with open('./log/test_log.txt','at') as file:
                file.write('%d-->test accuracy: %.2f%%\n'%(step,(accuracy*100)))
    
    #保存train loss, train accuracy, valid loss和valid accuracy    
    np.save('./log/train_loss.npy',np.array(train_loss))
    np.save('./log/train_accuracy.npy',np.array(train_accuracy))
    np.save('./log/valid_loss.npy',np.array(valid_loss))
    np.save('./log/valid_accuracy.npy',np.array(valid_accuracy))

 
def plot():
    '''
    @description: 画出训练过程中的训练集的loss，accuracy曲线和验证集的loss，accuracy曲线
    @params: 
    @return: 
    '''
    train_loss = np.load('./log/train_loss.npy')
    train_accuracy = np.load('./log/train_accuracy.npy')
    valid_loss = np.load('./log/valid_loss.npy')
    valid_accuracy = np.load('./log/valid_accuracy.npy')
    fig1 = plt.figure(1)
    axes1 = fig1.add_subplot(1,2,1)
    axes1.plot(train_loss,color='g')
    axes1.set_title('train loss')
    axes1 = fig1.add_subplot(1,2,2)
    axes1.plot(train_accuracy,color='m')
    axes1.set_title('train accuracy')
    plt.show()
    
    fig2 = plt.figure(2)
    axes2 = fig2.add_subplot(1,2,1)
    axes2.plot(valid_loss,color='green',linestyle='solid',marker='o')
    axes2.set_title('valid loss')
    axes2 = fig2.add_subplot(1,2,2)
    axes2.plot(valid_accuracy,color='m',linestyle='solid',marker='o')
    axes2.set_title('valid accuracy')
    plt.show()
           
if __name__ == '__main__':
    train('./data/',batch_size=50,learning_rate=0.1,max_iteration=50000)
   
 