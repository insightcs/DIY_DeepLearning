# 程序运行说明

---

## 1. 文件说明
<pre>
FaceIdentification/
├── data 存放划分的数据集对应的txt文件（存放每张图片的path name和label）
│   ├─ train_list.txt 训练集对应的txt文件
│   ├─ valid_list.txt 验证集对应的txt文件
│   ├─ test_list.txt 测试集对应的txt文件
├── log
│   ├── vgg 存放VGGNet训练过程中输出的log
│   │   └── vgg16.png VGG16的网络结构图
│   ├── resnet 存放ResNet训练过程中输出的log
│   │   └── resnet50.png ResNet50的网络结构图
├── model 
│   ├── resnet 存放训练好的ResNet模型
│   └── vgg 存放训练好的VGGNet模型
├── __int__.py   
├── load_data.py 划分数据集，从数据集中读取batch_size大小的数据
├── utils.py 计算模型预测的loss和准确率， 定义网络优化算法
├── vgg.py 构建VGG-Net16网络结构
├── vgg_train.py 训练VGG16网络模型，保存网络模型，测试模型预测准确率
├── res_net.py 构建ResNet50网络结构
└── resenet_train.py 训练ResNet50网络模型，保存网络模型，测试模型预测准确率
</pre>

## 2. 程序运行需要的平台
- python 3.5.2
- numpy 1.13.1
- pillow 3.3.1
- tensorflow-gpu 1.4.0:

## 2.程序运行说明
- 划分数据集：
	- webface数据集（人脸图片已经对齐）比较大，我已经上传到百度云盘中了，需要的话可以下载，链接：https://pan.baidu.com/s/1gf4bPXh 密码：z5k8。
	- 先修改load_data.py中以下程序
	- 然后命令行输入：python load_data.py，划分数据集
```python
if __name__ == '__main__':
    split_datasets(file_dir='../webface/',saved_dir='./data/')   #file_dir为人脸图片数据集所在的文件夹（比如我的数据集存放在"../webface/"文件夹下），saved_dir为划分之后的数据集对应的txt（存放每张图片的path name和label）文件，这个按照默认就行
```
- 训练VGG16网络模型：
python vgg_train.py
- 训练ResNet50网络模型：
命令行输入：python resnet_train.py
如需修改参数，可按照vgg_train.py和resnet_train.py中的示例程序修改，参数都有说明。
> **注**: 文件夹内所有程序均在Ubuntu 14.04中调试通过，只要安装必要的运行平台，程序可正确运行。