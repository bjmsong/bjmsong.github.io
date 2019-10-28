---
layout:     post
title:      deeplearning.ai之一
subtitle:   Neural Networks and Deep Learning
date:       2019-09-16
author:     bjmsong
header-img: img/dl/dl.png
catalog: true
tags:
    - 深度学习
---

### week1. 深度学习概论
- 深度学习是新时代的“电力”
- 驱动深度学习发展的因素
    - data
    - computation
    - algorithms

### week2. 神经网络基础
2.1 符号说明
- X：(nx,m), 一列是一个样本
    - nx个维度/特征（例如图像：64*64*3），m个样本
- Y:(1,m)

2.2 逻辑回归
- sigmoid function：将值转换到(0,1)区间
- 损失函数(loss function)
    - 衡量在单个样本上的表现
    - 要用凸函数，不然会得到很多个局部最优解( 所以不能用平方误差：$\frac{1}{2}(y^--y)^2$ ),而要用交叉熵：
    > $$L = - (ylog(y^-)+(1-y)log(1-y^-))$$ 
    
    验证：
    y=1，L= -logy^- => y^-越大（接近1）损失函数越小
    y=0, L= -log(1-y^-) => y^-越小（接近0）损失函数越小
    - $y^-$是预测值，y是真实值
    - 损失函数越小越好
- 成本函数（cost function）
    - 衡量在全体训练样本上的表现
    > $$J(w,b) = \frac{1}{m}\sum_i^m(L(y-^i,y^i))) $$ 
    - 推导：最大似然估计（给定样本的观测值概率最大）
        https://mooc.study.163.com/learn/2001281002?tid=2001392029#/learn/content?type=detail&id=2001702014
- 梯度下降法(Gradient Descent)
    - 参数初始化
    - 参数朝梯度下降的方向迭代，使得cost function朝全局最优解收敛 

 2.3 导数   
 - 即斜率
 - 计算图(Computation Graph)
    - 链式法则（chain rule）

2.4 向量化(vectorization)
- 即：将向量当作标量来进行运算
    - 并行计算？
- for循环很低效，效率差百倍
- np.dot(w,x)，np.exp(v),np.log(v),np.zeros(n,1),np.abs(v)...
- GPU，CPU 都有并行化指令(SIMD),numpy可以利用并行化，尽量使用numpy的内置函数计算

2.5 python广播(Broadcasting)
- 计算时将维度较少的变量自动展开

2.6 python/numpy,debug技巧
- 定义向量时
don't use: ```np.random.randn(5)```
use:```np.random.randn(5,1) or np.random.randn(1,5)``` 
- 通过assert 确定使用的是n×1、1×n的矩阵
- np.sum(...,keepdims=Ture):防止输出秩为1的数组

### week3. 浅层神经网络
3.1 概念
- input layer （第0层）
- hidden layer
- output layer

3.2 向量化
- 同一层的参数
- 多个样本

3.3 激活函数
- 为什么需要激活函数
    - 如果没有非线性激活函数，输出就是输入的线性函数
    - 输出层是唯一可以用线性激活函数的地方
- sigmod
    - 一般只用来做输出层的激活函数（0/1分类）
- tanh
    - 使数据中心化：平均值为0
    - 优于sigmod
- ReLU
    - 主流
    - 会使得梯度下降速度加快
    - 改进版：leaky ReLU，取负数时斜率不为0

3.4 反向传播算法(BP)
- 前向传递输入信号直至输出产生误差，反向传播误差信息更新权重矩阵
- 梯度下降在链式法则中的应用

3.5 随机初始化
- 不要全零
- Xaiver initialization

### week4. 深层神经网络
4.1 核对矩阵的维数
W[l] = (n[l],n[l-1]) , l:layer number,n:units in layer
b[l] = (n[l],1)
dW[l] = (n[l],n[l-1])
db[l] = (n[l],1)

Z[l],A[l],dZ[l],dA[l] = (n[l],m)

4.2 why “deep” works well？ 
- 类似于大脑工作原理 
    - 前几层能学习一些低层次的简单特征
    - 后几层把简单特征结合起来
- 电路理论
    - There are functions you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute.

4.3 参数和超参数
Parameters：W，b
Hyperparameters： learning rate,iterations,hidden layer number,choice of activation function，mini batch size，regulation 。。。
- 超参数可以控制参数
- 即使是对于同一个项目，最优参数可能会随着时间改变，需要通过试验决定










