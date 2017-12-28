# 程序运行说明：
-----

## 1. 文件说明
- ./data/：文件夹中存放数据集包括训练数据，测试数据。
- ./data/handwriting_digits_images/：文件夹中存放手写数字图片，总共10k张图片，图片的名字为其标记。
- ./log/：文件夹中存放模型训练过程中输出的log以及train loss和train accuracy曲线图。
- ./model/：文件夹中存放已经训练好的模型，只需要加载即可使用。
- ./nn_ops.py：为神经网络模型的基本运算操作，包括卷积，池化，relu，softmax等。
- ./lenet.py：为搭建的LeNet卷积神经网络模型。
- ./load_data.py：为加载数据的脚本文件。
- ./train.py：训练网络模型脚本文件，在命令行下直接执行即可。
- ./test.py：测试网络模型脚本文件，在命令行下直接执行即可。
- ./inference.py：为预测图片样本标记的脚本，为其传入样本所在路径，在命令行下执行即可。
## 2. 文件运行所需平台：
- Python 3.5
- numpy
- pillow: 读取图片
- pickle: 保存和加载模型
- gzip: 解压数据集
> **注**: 文件夹内所有程序均在Ubuntu 14.04中调试通过，只要安装必要的运行平台，程序可正确运行。

## 3. 网络模型训练结果
- 网络结构
![@architecture | center](https://github.com/insightcs/DeepLearning/blob/master/mnist/log/architecture.jpg)
- 数据集划分： 
  - 训练集train_dataset: 59500个样本
  - 验证集valid_dataset: 500个样本
  - 测试集test_dataset: 10000个样本
- 训练过程，每次迭代时，batch_size为50，每迭代一次耗时25秒左右，总共迭代20000次完成训练。
- 准确率：经过测试，在测试集上的最高准确率可达98.19%。
- 预测速度：预测10k个图片，总共耗时276.556秒，每个样本耗时0.028秒。
