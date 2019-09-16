---
layout:     post
title:      【笔记】deeplearning.ai之五
subtitle:   Natural Language Processing
date:       2019-09-16
author:     bjmsong
header-img: img/dl/dl.png
catalog: true
tags:
    - 深度学习
---

## week1.循环序列模型
1.1 词表示
- 词袋
- w2v 

1.2 RNN
- DNN为什么不能解决？
    - inputs，outputs can be different lengths in different examples
    - doesn't share feture learned from different positions of text
- RNN
    - use feature from different positions of text
    - tanh : common choice
- different types of RNN
    - many to many ：翻译
        - 输入长度 = 输出长度 ： Sequence to Sequence
    - many to one : 情感分类
    - one to many : 音乐生成,文本创作
- 《An Empirical Exploration of Recurrent Network Architectures》

1.3 language model
- 计算给定语句的概率：P(w1)*P(w1|w2)*P(w1,w2|w3)*....
    - 非常适合用RNN来建模
- build model
    - word-level,character-level
    - 构建语料库
    - 分词，并将词对应到字典(Tokenize)
    - RNN model : y -- 这个位置是这个词的概率

1.4 sampling novel sequences
- sampling a sequence from a trained RNN
    - 对每一次输出进行随机采样
- 应用：文本生成
- Vocabulary level : widely used
- Character level ： 序列会比较长，计算更耗时

应用：文本生成
    - 应用：写文章（诗，散文，周杰伦歌），起名字
    - char-rnn作者原文：http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    - keras实现写文章：https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
    - https://github.com/karpathy/char-rnn
    - https://www.jianshu.com/p/50fd465ec1e1
    - https://applenob.github.io/rnn_1.html
    - https://www.msra.cn/zh-cn/news/features/ruihua-song-20161226

1.5 Vanishing gradients in RNN 
- 很难有长期记忆
- 梯度爆炸的问题容易被发现
    - 可以用梯度修剪(clipping)解决：给梯度设置一个阈值范围 
- 梯度消失问题
    - LSTM,GRU

1.6 GRU：Gated Recurrent Unit
- C = memory cell
- 更新门：决定何时更新C 
- 视觉门，相关门
- simplier than LSTM

1.7 LSTM
- more powerful
- forgate gate,update gate,output gate

学习资料：
- 《Understanding LSTM network》 from Chris Ola的博客
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- https://stats.stackexchange.com/questions/205635/what-is-the-intuition-behind-a-long-short-term-memory-lstm-recurrent-neural-ne
- http://blog.echen.me/2017/05/30/exploring-lstms/

1.8 Bidirectional RNN

1.9 Deep RNN


## week2.自然语言处理与词嵌入
2.1 Word Embedding
- 训练模型需要一个庞大的语料库
- 直接拿网上pre-train好的模型
- 应用
    - 命名实体识别、文本摘要、文本解析、指代消解
    - 应用少一些：语言模型、机器翻译
        - 这些问题本身数据量已经比较大了，不需要transfer learning
- 特性：help with analogy reasoning
    - man-woman ~~ king-queen
- 词嵌入矩阵E
- NNLM
    - 输入上下文，预测目标词
    - fixed window：比如前4个单词预测后一个单词
- word2vec
    - 《Efficient estimation of Word Representation in vector space》
    - skip-gram
        - 随机选择上下文和目标词配对
            - X:context word 
                - sample
              y:target word
              model:softmax function  
                - 缺点：计算效率低
                    - 改进(树状结构)：分级 softmax
              loss function:cross-entropy 
    - CBOW
        - 用两边的词预测中心的词
    - Negative sampling
        - 《Distributed Representations of Words and Phrases and their Compositionality》
        -  解决skip-gram算法softmax计算效率低的问题
        - x：采样得到上下文词和目标词对
          target：这两个词是否在同一个上下文中
          model：sigmoid function
    - Glove
        - 《GloVe: Global Vectors for Word Representation》
- debiasing word embedding
    - 《Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings》
    - 性别、种族、年龄等方面的歧视
- 中文词向量pre-trained：
https://ai.tencent.com/ailab/nlp/embedding.html
https://github.com/Embedding/Chinese-Word-Vectors

2.2 情感分类
- RNN：many to one

## week3.序列模型和注意力机制
3.1 Sequence to Sequence Learning
- Encoder-Decoder
- pick the most likely sentence
    - machine translation ~~ conditional language model
    - difference:
        - language model : choose randomly
        - machine translation : choose the best
    - Beam Search
        - 预测下一个词时：挑选概率最大的N个词
- 应用
    - 机器翻译
    - 文章摘要
    - 语音识别
    - 图像描述
        - Encoder：CNN提取图像特征
        - Decoder：RNN生成描述语句
        《Deep captioning with multimodal recurrent neural networks》
        《Show and tell:Neural image caption generator》
        《Deep visual-semantic alignments for generating image descriptions》
- 学习资料
    《Sequence to Sequence Learning with Neural Networks》
    《A Critical Review of Recurrent Neural Networks for Sequence Learning》

3.2 Bleu score
- 评估文本生成的记过：机器翻译、图像描述等
- 计算机器翻译结果在人类翻译结果中的重合比例
- 实现方式open-sourced
- 《A method for automatic evaluation of machine translation》

3.3 序列标注(Sequence Tagging)
- 应用场景
    - 分词
    - 词性标注
    - 命名实体识别
    - 关键词抽取
    - 词义角色标注
- 算法
    - HMM
    - 最大熵模型
    - CRF
    - RNN
        - Bi-LSTM+CRF
- 学习资料
    - Bidirectional LSTM-CRF MOdels for Sequence Tagging

3.4 Attention
- widely used in deep learning
- 常见结构
    - 双向RNN
    - 注意力权重
        - 通过一个小型神经网络学习得到
    - RNN 
- 《Neural Machine Translation By Jointly Learning to Align and Translate》
  《Show attention and tell:neural Image Caption Generation with Visual Attentio》
https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/
https://blog.csdn.net/xiewenbo/article/details/79382785

3.5 语音识别

3.6 触发字识别
- 