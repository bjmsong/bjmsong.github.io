---
layout:     post
title:      Transformer
subtitle:   
date:       2022-09-01
author:     bjmsong
header-img: img/transformer/logo.jpg
catalog: true
tags:
    - 深度学习
---

## 自注意力机制（self-attention）

很多问题的输入是一组向量，并且向量的数量可能不一样，例如nlp、语音识别、graph等。


<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/1.png) 
</li> 
</ul> 


输出可能跟输入数量一样（Sequence Labeling），也可能不一样（Seq2Seq）。

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/2.png) 
</li> 
</ul> 


我们通常需要考虑输入数据的上下文，才能使网络更好地理解输入数据。传统的做法是固定一个window，把一个window的输入输入同一个网络中。但是window太大会使得计算量太大，window太小上下文不完整。

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/3.png) 
</li> 
</ul> 

因此提出了self-attention机制，输入经过self-attention转换，获得了上下文信息。

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/4.png) 
</li> 
</ul> 


<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/5.png) 
</li> 
</ul> 

计算输入之间的关联性（即`attention score`）的方法有`Dot-Product`、`Additive`

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/6.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/7.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/8.png) 
</li> 
</ul> 

`q，k，v`可以通过矩阵乘法得到

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/9.png) 
</li> 
</ul> 

`attention score`也可以通过矩阵乘法得到

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/10.png) 
</li> 
</ul> 



`b`可以通过矩阵乘法得到

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/11.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/12.png) 
</li> 
</ul> 

用公式表示就是：

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/13.png) 
</li> 
</ul> 

Q和K进行点积后，除以$$\sqrt{d_k}$$, 这样可以避免不同输入点积的结果差异过大，导致softmax趋近于1和0。这样会导致梯度太小，模型参数训练缓慢。

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/14.png) 
</li> 
</ul> 



### Multi-head Self-attention

相关性可以有很多种，模拟卷积神经网络多输出通道的效果

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/15.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/16.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/17.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/18.png) 
</li> 
</ul> 

### Positional Encoding

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/19.png) 
</li> 
</ul> 

### self-attention vs CNN

CNN是self-attention的特例，CNN的问题：相隔较远的时序信息很难关联

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/20.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/21.png) 
</li> 
</ul> 

self-attention需要更多的训练数据

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/22.png) 
</li> 
</ul> 



### self-attention vs RNN

RNN的问题：无法并行效率低，相隔较远的时序信息很难关联

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/23.png) 
</li> 
</ul> 



## Transformer

### Seq2seq问题应用广泛

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/24.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/25.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/26.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/27.png) 
</li> 
</ul> 



### Encoder

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/28.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/29.png) 
</li> 
</ul> 

Residual：残差连接

Feed Forward：全连接网络

Layer Normalization和Batch Normalization都是深度神经网络中常用的归一化方法，用于减少梯度消失和梯度爆炸等问题。它们的主要区别在于归一化的对象不同。

Batch Normalization在训练阶段计算同一个batch内**同一个特征**的均值和方差，在预测阶段计算所有样本**同一个特征**的均值和方差，然后进行归一化。

Layer Normalization在训练阶段计算同一个batch内**同一个样本**的均值和方差，在预测阶段计算**同一个样本**的均值和方差，然后进行归一化。。

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/30.png) 
</li> 
</ul> 



### Decoder

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/31.png) 
</li> 
</ul> 

Decoder有两种模式：Autoregressive(AT)，Non-Autoregressive(NAT)

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/32.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/33.png) 
</li> 
</ul> 

masked self-attention：因为decoder是按顺序产生的，在计算输出的时候要把之后的数据屏蔽掉

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/34.png) 
</li> 
</ul> 

Cross-attention

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/35.png) 
</li> 
</ul> 

k，v来自encoder，q来自decoder

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/36.png) 
</li> 
</ul> 



### Training

训练阶段，Decoder的输入是真实数据（Ground Truth）

adam，正则化(dropout，label smoothing)

Transformer可调的参数不多

attention对数据的假设更少，需要更多的数据，更大的模型才能训练出好的结果（对比RNN，CNN）:  越来越贵

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/37.png) 
</li> 
</ul> 

有一些training tips：

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/38.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/39.png) 
</li> 
</ul> 


<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/40.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/41.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/42.png) 
</li> 
</ul> 

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/transformer/43.png) 
</li> 
</ul> 
