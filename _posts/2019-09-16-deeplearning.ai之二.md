---
layout:     post
title:      deeplearning.ai之二
subtitle:   Improving Deep Netural Networks
date:       2019-09-16
author:     bjmsong
header-img: img/dl/dl.png
catalog: true
tags:
    - 深度学习
---


### week1 深度学习的实用层面
1.1. train/dev/test
- 数据量如果很大，可以降低dev/test的数据比例。例如：98%/1%/1%
- 验证集和测试集should come from same distribution
  - **验证集上的数据分布和预测目标应该和最终应用场景一致**
- 也可以不要测试集（？）

1.2 Bias and Variance

- 如何判断bias、variance
  - 假设最优误差是0% ：一般用人类误差水平作为替代
  - 训练集误差1%，验证集误差15% ： high variance
  - 训练集误差15%，验证集误差16% ： high bias
  - 训练集误差15%，验证集误差30% ： high bias,high variance  （欠拟合部分数据，过度拟合部分数据）
  - 训练集误差0.5%，验证集误差1% ：low bias,low variance
- High Bias:  under fitting -- bigger network，more iterations
- High Variance: over fitting -- more data,regularization
- **深度学习时代，bias-variance trade-off经常不再是一个问题，可以在降低bias的同时不增大variance，反之亦然**

1.3 正则化
- 降低variance
- L1正则可以起到特征选择的作用 （参数 will be sparse）
- L2正则在深度学习中使用更多
  - L2正则也叫“权重衰减”(weight-decay):梯度下降过程中参数w会越来越小
- dropout： 对于每一个样本，都以一定概率消除网络节点，以一个精简的网络结构来训练模型
  - 实现：Invertedd dropout（反向随机失活）
  - **只用于训练阶段，预测阶段：no dropout**
  - 因为输入随时可能被消灭，因此drop out起到了降低权重的作用
  - 不同层的保留概率(keep-prop)可以不同
  - 主要用于计算机视觉领域：数据不足，容易过拟合
  - 缺点：cost function 缺乏明确的定义
- 其它降低过拟合的方法
  - 增加数据:图片翻转、旋转、剪裁
  - early stopping ： 当验证集误差不再减小，提前终止训练
    - 训练集误差会一直减小下去

1.4 输入标准化(Normalizing)
- 训练集和测试集要一起归一化
- 优点：
  - 不同特征值范围类似，代价函数会更圆一些：提升学习速度
  - 不同特征对结果做出的贡献相同：提高精度
- 标准化方法
  - min-max normalization
  - z-score normalization
    - step1：减去平均值  -- 均值变成0
    - step2：除以方差和  -- 方差变成1
  - log转换

1.5 梯度消失、梯度爆炸
- 导数指数式增大、减小
- 层数越多，问题越严重
- 解决办法：初始化权重使得权重不是太大，也不是太小
  - 如对于ReLU：np.random.randn(shape)*np.sqrt(2/n)
  - 如对于tanh：Xavier initialization
- LSTM解决的是梯度消失问题

1.6 梯度的数值逼近
- 双边误差比单边误差更精确



### week2 优化算法


2.1 mini-batch

- 随着迭代的进行，cost function会有波动，不过趋势是减小的  
- choose mini-batch size：64~512
- if training set is small ：use batch gradient descent
  

2.2 指数加权平均
$$
v_t=\beta*v_{t-1}+(1-\beta)\theta_t
$$

$$
当\beta=0.1，近似等于计算最近10天的平均值，优点是占内存很小
$$

- bias correction(偏差修正)：初期估算不准
  $$
  v_t -> \frac{v_t}{(1-\beta^t)}
  $$
  

2.3 Momentum
$$
v_{dw}=\beta*v_{dw}+(1-\beta)dw
$$

$$
w:=w-\alpha*v_{dw}
$$

- 平均之后，纵轴方向相互抵消：减缓梯度下降波动的幅度
- 并不是对所有cost function都有效：对碗状的cost function有效

$$
\beta=0.9 is good
$$

2.4 RMSprop:root mean square prop
- 减缓纵轴方向的学习，加快横轴方向的学习

- 可以使用更大的学习速率
  $$
  s_{dw}=\beta*s_{dw}+(1-\beta)dw^2
  $$

  $$
  w:=w-\alpha*\frac{dw}{\sqrt{s_{dw}}}
  $$

2.5 Adam:Adaptive Moment Estimation
- 结合Momentum和RMSprop
  $$
  v_{dw}=\beta1*v_{dw}+(1-\beta1)dw
  $$

  $$
  s_{dw}=\beta2*s_{dw}+(1-\beta2)dw^2
  $$

  $$
  v_{dw} = \frac{v_{dw}}{(1-\beta1^t)}
  $$

  $$
  s_{dw} = \frac{s_{dw}}{(1-\beta2^t)}
  $$

  $$
  w:=w-\alpha*\frac{v_{dw}}{\sqrt{s_{dw}}+\epsilon}
  $$

  $$
  \beta1=0.9,\beta2=0.999,\epsilon=10^-8  is recommended
  $$

2.6 Learning rate decay

- 随着迭代的进行，降低学习速率

2.7 局部最优和鞍点
- 在海量维度的情况下，很难达到局部最小点 
  - **不用担心梯度下降会困在局部最优**
  - **低维空间的直觉不一定可以应用到高维度空间**
  - 更有可能到达鞍点
    - 梯度为零
    - 一些维度向上，一些维度向下
    - 鞍点附近学习速率慢
      - Momentum、RMSprop、Adam可以加快学习



### week3 超参数调试、Batch Normalization和程序框架

3.1 超参数调试
- 超参数重要性排序
  - **最重要：alpha：learning rate**
  - **次重要**
    - beta:momentum
    - mini-batch size
    - number hidden units
  - **次次重要**
    - number of layers
    - learning rate decay
  - **基本不用调**
    - beta1，beta2：Adam
- 如何调参
  - **try random values, don't use grid**
    
    - 不同超参数的重要性不一样
    
  - 若参数范围是 [0.0001,1],如果只是random，90%参数会集中在[0.1,1]的范围，可以采用以下方式：
      
      ```python
      r = -4 * np.random.randn()
      a = 10^r
      ```
      
  - 若参数范围是 [0.9,0.999]
      
      ```
      r random in [-3,-1]
      b = 1 - 10^r
      ```
- **coarse to fine:先粗调后细调**
- 超参数要定期重新调整
- 调参模式
  - babysitting one model
    - 资源不够
    - 照看一个模型
    - 根据表现调整参数
  - training many models in parallel
    - 机器资源够多
    - 同时训练不同组的超参数
- 不同领域的想法可以互相借鉴
  
  - NLP,Vision,Speech,Ads,Logistics



3.2 **Batch normalization**

- https://www.cnblogs.com/guoyaohua/p/8724433.html

$$
z_{norm} = \frac{z-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

$$
z^{bn}=\gamma*z_{norm}+\beta
$$

- 隐藏层输入(z)做标准化
- 先归一化，再重新分布
- 优点：加速学习
- 为什么有效？
    - learning on shifting input distribution
        - reduce covariate shift(输入样本分布发生变化) ： fix mean and variance
    - slight regulariza tion effect ：add　some noise to each hidden layer's activations
- 如何在测试集应用BN 
    - 训练集：**在每个mini-batch单独应用BN**
    - 测试集：使用训练集每个mini-batch的mean&variance 做指数加权平均 作为mean&variance的估算

3.3 softmax
- 将逻辑回归推广到多分类
   
   - C=2就是逻辑回归
   
- 线性激活函数

- loss function
  $$
  L = -\sum_i^C ylog(y^-)
  $$
   



### 参考资料

- https://zhuanlan.zhihu.com/p/32230623
- 《An overview of gradient descent optimization algorithms》