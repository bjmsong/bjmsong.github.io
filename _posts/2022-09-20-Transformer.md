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

## https://bjmsong.github.io/2020/03/03/%E4%BB%8EWord-Embedding%E5%88%B0Bert%E6%A8%A1%E5%9E%8B-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E4%B8%AD%E7%9A%84%E9%A2%84%E8%AE%AD%E7%BB%83%E6%8A%80%E6%9C%AF%E5%8F%91%E5%B1%95%E5%8F%B2/

## https://www.bilibili.com/video/BV1Di4y1c7Zm/?is_story_h5=false&p=1&share_from=ugc&share_medium=android&share_plat=android&share_session_id=8ff19c57-bbed-49d2-8fa2-2cd31b573fcf&share_source=WEIXIN&share_tag=s_i&timestamp=1665311648&unique_k=dbwe0H3


## 
https://github.com/wangshusen/DeepLearning
    - 第五节

##
- 目前最强的特征抽取器，替代RNN,可并行
- 资料
  - **https://jalammar.github.io/illustrated-transformer/ （解释得很清楚）**
  - **http://nlp.seas.harvard.edu/2018/04/03/attention.html （代码 !）**
  - github: attention-is-all-you-need-pytorch 
  - https://github.com/ongunuzaymacar/attention-mechanisms
  - https://zhuanlan.zhihu.com/p/48508221
  - paper：
      - Attention is All you need (不好懂)
      - Netural Machine Translation by Jointly Learning to Align and Translate
- Seq to Seq 结构
- Stacked encoder + Stacked decoder
- Encoder：
    - **self-attention**
        - 基本步骤
            - first step ： calculate “query”, “key”, and “value” vectors by Multiple Query,Key,Value Matrix
            - second step : calculate a score 
                - score each word of the input sentence against this word
                - dot product of the query vector with the key vector
            - third step : divide the score 
                - leads to have more stable gradients
            - forth step : softmax
            - fifth step : multiply each value vector by the softmax score
            - sixth step : sum up the weighted value vectors
                -  output the self-attention layer at this position
        - multi-head attention ： improve performance
        - Representing The Order of The Sequence Using Positional Encoding 
    - feed forward(非线性转换)
- Decoder
    - self-attention
    - encoder-decoder attention
    - feed forward
- 几种不同的attention
    - self-attention
        - pair-wise
        - bi-directional
    - decoder-attention
        - 只跟前几个相关
        - uni-directional
    - encoder-decoder attention

    
## 开源实现
tensor2tensor

## https://github.com/huggingface/transformers


