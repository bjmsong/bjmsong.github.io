---
layout:     post
title:      deeplearning.ai之四
subtitle:   Convolutional Neural Networks
date:       2019-09-16
author:     bjmsong
header-img: img/dl/dl.png
catalog: true
tags:
    - 深度学习
---
### week1.卷积神经网络
1.1 卷积(kernel/filter/convolution)

- 作用
    - 提取图像特征,从浅层到深层：如边缘检测 
    - 参数共享：图像维度太多，传统DNN参数太多，容易过拟合
- 卷积：用卷积核跟图像像素进行乘法运算
    - 真正在数学中称作：“互相关”
- 卷积核的值可以通过神经网络学习得到
- 卷积核维度通常是奇数：3`*`3,5`*`5

1.2 padding

- 沿着图像边缘，填充像素
- n`*`n维图像，经过f`*`f卷积后
    - without padding，变为 (n-f+1)`*` (n-f+1)维图像
    - with padding(p)，变为 (n+2p-f+1)`*` (n+2p-f+1)维图像
- 原因
    - 图像经过卷积后会越来越小
    - 边缘的点被卷积的次数太少，信息丢失
- 两类卷积：
    - Valid：不做padding
    - Same：Pad so that output size is the same as the input size
        - p=(f-1)/2

1.3 卷积步长(Stride)
- 卷积核移动的步长
- 卷积后的维度：1+(n+2p-f)/s `*` 1+(n+2p-f)/s

1.4 当图像是多通道的情况(如：RGB)
- 深度：depth
- 卷积核也是多通道的，通道数量和图像一致
- 卷积后的结果是单通道的：把多个通道的结果汇总
- 可以有多个卷积核(分别提取图像不同的特征)，这样卷积的结果也是多通道的

1.5 单层卷积神经网络
- 多层卷积之后：图像维度越来越小，通道数越来越多 

1.6 池化层
- 缩减模型大小，调高运算速度，提高选取特征的鲁棒性
- max pooling
- no parameters to learn

1.7 为什么卷积层有效
- **Parameter sharing(参数共享):**A feature detector (such as a vertical edge detector) that’s useful in one part of the image is probably useful in another part of the image.
- **Sparsity of connections(稀疏连接)**：In each layer, each output value depends only on a small number of inputs.



### week2.深度卷积网络：实例探究

2.1 经典网络结构
- LeNet-5
    - 《Gradient-based learning applied to document recognition》
- AlexNet
    - 《ImageNet classification with deep convolutional neural networks》 
- VGG-16
    - 《Very deep convolutional networks for large-scale image recognition》
- ResNet：152层

2.2 残差网络
- ResNet 
- skip connection：可以把信息传到很远的层，解决了梯度消失/梯度爆炸，网络不能太深的问题

2.3 `1*1`卷积核  

- 降维

2.4 Inception 
- 把不同参数的卷积层，pooling层都放到一起，计算
- 计算代价大 
    - 1`*`1 卷积核可以减少计算
- 《Going deeper with convolutions》

2.5 数据增强
- mirroring
- random cropping
- rotation
- color shifting

2.6 计算机视觉现状
- 数据量越少，越依赖手工特征
- 数据量越多，越可以使用深度学习
- 提升精度的方法（在工业界并不实用）
    - assemble
    - multi-corp
- use open source

### week3.目标检测




### week4.特殊应用：人脸识别和神经风格转换




- 《Visualizing and Understanding Convolutional Networks》
