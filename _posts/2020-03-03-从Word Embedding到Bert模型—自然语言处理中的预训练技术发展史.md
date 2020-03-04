---
layout:     post
title:      从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史 
subtitle:   
date:       2020-03-03
author:     bjmsong
header-img: img/nlp/nlp.jpg
catalog: true
tags:
    - NLP

---



### 预训练（Pre-training） 
- 借鉴于图像领域
- 不同方式  
    - Fine tunning 
    - Frozen
- transfer learning
- NLP：word2vec，ELMO，Bert



### Language Model

<ul> 
<li markdown="1"> 
![]({{site.baseurl}}/img/nlp/语言模型.jpg) 
</li> 
</ul>

- 每个词独热表示，词之间无法表示相似性
- measures how likely a valid sentence c
    - P(w1,w2,,,wd) = P(w1)*P(w2|w1)*P(w3|w1,w2)*...*P(wd|w1,w2...wd-1)
    - N-gram：假设第N个词只跟前N-1个词有关
        - N=2：P(w1,w2,,,wd) = P(w1)*P(w2|w1)*P(w3|w2)*...*P(wd|wd-1)
        - 每个概率以通过直接从语料中统计2个词同时出现的次数得到：需要大量训练语料
- LSTM 非常适合来建模
- https://zhuanlan.zhihu.com/p/32829048



### Word Embedding

- 《Efficient Estimation of Word Representations in Vector Space》
- Distributed Representation
    - distributed vectors
    - 获得语义上的相似度 semantic similarity
- Some Good Word Embedding
    - NNLM（Netural Network Language Model）
        
        - 语言模型
        
    - 只考虑前面的n个单词
      
      <ul> 
      <li markdown="1"> 
      ![]({{site.baseurl}}/img/nlp/NNLM.jpg) 
      </li> 
      </ul>
      
    - Word2Vec：CBOW,SkipGram 
    
      <ul> 
      <li markdown="1"> 
      ![]({{site.baseurl}}/img/nlp/word2vec.jpg) 
      </li> 
      </ul>
      
        - 也是做语言模型任务
        - CBOW : 根据前后的几个单词，预测中心单词
        - SkipGram：根据中心单词，预测前后的单词
      
    - Glove
    
    - 缺点：    
        - not distinguish contextualized (同一个词，不同上下文都一样的)
        - does not capture long-term dependency
        - Shallow Model，could not learn hierarchical representation (层次结构的表示)
            - 例如：深度图像模型，每一层能学到不同的特征
- Better Ways:Deep Model,contextualized,语言模型
    - ELMO:LSTM
    - Bert:Transformer



### ELMO

- 《Deep contextualized word representation》
- 根据上下文动态调整单词的Word Embedding表示
- 两阶段过程
- stage 1：利用**双向**语言模型进行预训练
    - Stacked-biLSTM
    - 得到三个Embedding
        - 单词的Word Embedding ： 单词特征
        - 第一层LSTM中对应单词位置的Embedding ： 句法特征
        - 第二层LSTM中对应单词位置的Embedding ： 语义特征
- stage 2：
    - 提取对应单词的网络各层的Word Embedding 作为新特征补充到下游的任务中
        - 三个Embedding特征整合成一个Embedding
- 缺点
    - 没法并行   -- Transformer解决
    - 梯度消失：long term dependency  -- Transformer解决
    - 前向后向单独处理：伪双向  -- Masked Language Model解决



### Attention（注意力机制）

https://zhuanlan.zhihu.com/p/37601161
- soft attention
    - encoder-decoder
    - decoder阶段，分配不同输入位置的权重
    - 本质思想
        - 对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数
    - 三阶段计算过程
        - 根据Query和某个Key_i,计算两者的相似性
            - 点积
            - Cosine相似性
            - MLP网络
        - 数值转换
            - 引入类似Softmax计算方式
        - 加权求和
- self attention
    - Source内部元素之间 or Target内部元素之间
    - 更容易捕捉句子中长距离的相互依赖的特征
        - 优于LSTM



### Transformer

- 目前最强的特征抽取器，替代RNN,可并行
**https://jalammar.github.io/illustrated-transformer/ （解释得很清楚）**
**http://nlp.seas.harvard.edu/2018/04/03/attention.html （代码 !）**
https://zhuanlan.zhihu.com/p/48508221
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



### GPT

《Improving Language Understanging by Generative Pre-Training》
- 基于Fine-tunning的模式
- stage 1 : 利用**单向**语言模型进行预训练
    - Transformer作为特征抽取器
- stage 2 : 通过fine-tunning模式解决下游任务
    - 改造下游任务的网络结构要和GPT的网络结构一样
    - 利用第一步预训练好的参数初始化
    - 再用手头的任务训练网络，fine-tunning



#### GPT 2.0

https://zhuanlan.zhihu.com/p/56865533
- 改进
    - 第二阶段的fine-tunning换成了无监督



### Bert:Stacked Transformers Encoder

《Bert：Pre-training of Deep Bidirectional Transformers for Language Understanding》
- https://github.com/google-research/bert
- https://www.jiqizhixin.com/articles/2018-11-01-9
- openAI，Bert
    - Transformer Decoder
- 跟GPT采用完全相同的两阶段模型
- 区别
    - 预训练阶段采用**双向**语言模型
    - 语言模型数据规模更大 
- 如何改造下游任务
    - 可以做任何NLP的下游任务，具备普适性
- 如何用Transformer做双向语言模型？
    - Masked 语言模型
        - 类似CBOW的思路
- BERT模型从训练到部署全流程
https://mp.weixin.qq.com/s?__biz=MzU2OTA0NzE2NA==&mid=2247512760&idx=1&sn=3e81bb5de97958fd98c5544f960b4044&chksm=fc8653abcbf1dabd26a7a69dfbe02c9e2341762576c351ee472ccecc8dc80b18e2ebb9e5a937&mpshare=1&scene=1&srcid=&key=8a4c04f4ab18b28835525b12df33490e1d45f84da39debbc07cfd6b706579e9e409bab04a47089ab00b8d239cf5603c4e13ea8bf5b6752bfb5e1d58f89ce51b899e88b8a535c5639c66e1138754cd6b0&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+10&version=62060833&lang=en&pass_ticket=v4JZjGGHWR6FmQjR%2BtO4mgfPSwZ4kYbd%2F%2BG7e42FZcIQX1pyZQtKNnFtbFw7nNUO
https://github.com/xmxoxo/BERT-train2deploy
- 
https://mp.weixin.qq.com/s?__biz=Mzg5ODAzMTkyMg==&mid=2247486717&idx=1&sn=5b75ab53fdb9b1d1d1f175fe0e6be390&chksm=c06984a0f71e0db67f4088cae3b46435ad306f363c17e7ec50653fea176d26008e1033ed6893&mpshare=1&scene=1&srcid=&sharer_sharetime=1565741471599&sharer_shareid=602b1ccf63ca4ea52755ecd058f6d407&key=2ff8c7df758d8641ca53da8569f88b3d356857ed5bb8f8298a639855f2a1c54aff58bdfd33308d802628a1737da06d1b41bd56acbb6637bc3237a6501c3f5fa3a91b76d336b7e8173ca4b9c6226922d1&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+7&version=62060833&lang=zh_CN&pass_ticket=SQMKIlAXXYyyJnvPJjF6UKR9UIDp5ZqQuA%2FnQRDL0VNI68a5Mb3Z8v9wCIFR%2FvOc
https://github.com/CyberZHG/keras-bert
https://github.com/ymcui/Chinese-BERT-wwm
https://github.com/brightmart/albert_zh
https://www.jiqizhixin.com/articles/2018-11-01-9
https://zhuanlan.zhihu.com/p/46652512
https://zhuanlan.zhihu.com/p/68446772
- tinyBert
https://mp.weixin.qq.com/s?__biz=MzU2OTA0NzE2NA==&mid=2247519264&idx=3&sn=6613b053d03f8ae693b35f01f2612f7a&chksm=fc866d33cbf1e425d68be5aff6494b9cafcc8d077ffb74812e0f15bf14cdb2d08990856e0a30&mpshare=1&scene=1&srcid=&sharer_sharetime=1579346001073&sharer_shareid=7c5e66b1f9f5cbffe2ecd9a51d98b88e&key=1eff3ffcd7f051286401e8df28e053bfd1710bd82255a06bf00e20b02c0d3b8dba1ea3d17046062c007fb94cae199d97c0c1b00307399f814d19d1f633c66709f79243e8d550f8a185f722b053029335&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+10&version=6208006f&lang=en&exportkey=AVY%2BEePNjTIArx%2Fx%2BH%2B0fis%3D&pass_ticket=gm9%2BABipytZtikZMFSsOT%2BF8UWnvzj43lGKfYWCSYCmWpSwU4u2GX4poYw0bUAql



### XLNet

https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247491563&idx=1&sn=b139fae75de75cf1ecd78dca53007df1&chksm=fbd4ad87cca32491ba551414dc16ba1ca0260656f4d34a542c81d842f575b39cd1ddf013a7a5&mpshare=1&scene=1&srcid=0624V6Uh43gaGrOyXY3mzvAI&key=8a4c04f4ab18b288934eb3d56fb0199a0833136f6c3174390c79094a683f55dea1c40ab99af7989a8bd39122933421054ef5dc834f1eee3f545738f942539e144bd6c20d18f075cd9ea5c173246723e8&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+10&version=62060833&lang=en&pass_ticket=xDHREyRWT%2FIcmzT6U8K3y2sLJnSLl5%2BbcVHGIJW3d3SUi8E0sxYQmPUenTHpL7MQ



### 参考资料

- https://zhuanlan.zhihu.com/p/49271699
- https://zhuanlan.zhihu.com/p/47488095?utm_source=wechat_session&utm_medium=social&utm_oi=30249563717632&from=singlemessage
- https://zhuanlan.zhihu.com/p/66676144?utm_source=wechat_session&utm_medium=social&utm_oi=1114227199359782912&from=singlemessage&s_s_i=fg41NsREUSsyBOQAo%2Flk%2BHPO5qCB2tw7PkZHF1Yx0cA%3D&s_r=1
- 中文Bert预训练模型
https://github.com/ymcui/Chinese-BERT-wwm
- https://shomy.top/2017/07/28/word2vec-all/
- https://zhuanlan.zhihu.com/p/30302498
