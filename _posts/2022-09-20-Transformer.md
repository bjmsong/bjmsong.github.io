---
layout:     post
title:      Transformer
subtitle:   
date:       2022-09-20
author:     bjmsong
header-img: 
catalog: true
tags:
    - 深度学习
---
## 
https://github.com/wangshusen/DeepLearning
    - 第五节

## 李沐读paper
https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.999.0.0
- MLP、CNN、RNN之后第四大类基础模型
- 最开始应用在机器翻译(encoder-decoder)，后面被nlp几乎所有任务采用，作为预训练模型，跟CNN一样，不需要单独对每个任务做处理，Transformer可以通用。也被图片、语音等领域广泛采用，可以处理多模态的数据。
- 开源代码，扩大工作的影响力
- RNN的问题：无法并行效率低，相隔较远的时序信息很难关联（attention解决这个问题）
- CNN的问题：相隔较远的时序信息很难关联
- Layer norm：对每个样本做归一化（均值为0，方差为1）
    - batch norm：对每个特征做归一化
- Attention
    - query与key计算相似度（点击），作为权重与 v进行计算
        - 矩阵乘法的方式计算，可以并行  
    - mask: t时刻query只看t时刻之前的k-v pair
    - multi-head： 学到不同的模式
    - self-attention
- 残差连接
- point-wise Feed-Forward Network：MLP
- positional encoding：attention本身无法携带时序信息，把时序信息带到输入里面
- attention对数据的假设更少，需要更多的数据，更大的模型才能训练出好的结果（对比RNN，CNN）:  越来越贵
- 训练：adam，正则化(dropout，label smoothing)
- Transformer可调的参数不多
- 写文章技巧
    - 讲一个故事，为什么做，设计理念等
    - 重要的部分不要默认大家都知道，需要做介绍
    - 要写得让其它领域的人也能看懂

## https://github.com/huggingface/transformers


