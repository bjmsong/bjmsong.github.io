---
layout:     post
title:      推荐系统之
subtitle:   Youtube DNN
date:       2020-04-22
author:     bjmsong
header-img: img/Recommendation System/th.jpg
catalog: true
tags:
    - 推荐系统
---
## 算法架构

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/Recommendation System/youtubeDNN/架构.png) 
</li> 
</ul> 

- 第一层是`Candidate Generation Model`：完成候选视频的快速筛选，这一步候选视频集合由百万降低到了百的量级
- 第二层是用`Ranking Model`：完成几百个候选视频的精排


## 召回层
<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/Recommendation System/youtubeDNN/召回.png) 
</li> 
</ul> 

- 自底而上看这个网络，最底层的输入是用户观看过的video的embedding向量，以及搜索词的embedding向量
- 这个embedding向量是怎么生成的：作者是先用word2vec方法对video和search token做了embedding之后再作为输入的，这也是做embedding的“基本操作”
    - 当然，除此之外另一种大家应该也比较熟悉，就是通过加一个embedding层跟上面的DNN一起训练
- 特征向量里面还包括了用户的地理位置的embedding，年龄，性别等。然后把所有这些特征concatenate起来，喂给上层的ReLU神经网络
- 三层神经网络过后，是softmax函数。这里Youtube把这个问题看作为用户推荐next watch的问题，所以输出应该是一个在所有candidate video上的概率分布，自然是一个多分类问题
- 这里有一个问题是总共的分类有数百万之巨（备选video的数量），这在使用softmax训练时无疑是低效的。
    - YouTube的做法是：负采样（negative sampling）并用importance weighting的方法对采样进行calibration
    - https://www.tensorflow.org/extras/candidate_sampling.pdf
    - http://www.aclweb.org/anthology/P15-1001
- 在candidate generation model的serving过程中，YouTube不直接采用训练时的model进行预测，而是采用了一种最近邻搜索的方法
    - 这是一个经典的工程和学术做trade-off的结果，在model serving过程中对几百万个候选集逐一跑一遍模型的时间开销显然太大了，因此在通过candidate generation model得到user 和 video的embedding之后，通过**最近邻搜索的方法的效率高很多**。我们甚至不用把任何model inference的过程搬上服务器，只需要把user embedding和video embedding存到redis或者内存中就好了。


## 排序层
<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/Recommendation System/youtubeDNN/ranking.png) 
</li> 
</ul> 

- 引入另一套DNN作为ranking model的目的是：引入更多描述视频、用户以及二者之间关系的特征，达到对候选视频集合准确排序的目的
    - 排序层跟召回层比较重大的区别是特征方面
>During ranking, we have access to many more features describing the video and the user's relationship to the video because only a few hundred videos are being scored rather than the millions scored in candidate generation.    
- 从左至右的特征依次是
    - impression video ID embedding: 当前要计算的video的embedding
    - watched video IDs average embedding: 用户观看过的最后N个视频embedding的average pooling
    - language embedding: 用户语言的embedding和当前视频语言的embedding
    - time since last watch: 自上次观看同channel视频的时间
    - #previous impressions: 该视频已经被曝光给该用户的次数
- 优化目标：每次曝光预期播放时间（expected watch time per impression）
    - 没有采用经典的CTR，或者播放率（Play Rate）
- 采用了`weighted logistic regression`作为输出层


## code
https://github.com/shenweichen/deepmatch
https://github.com/wangkobe88/Earth

## 参考资料
- b站 水淼笔记
- Deep Neural Networks for YouTube Recommendations, 2016
- 王喆的三篇博客
    - https://zhuanlan.zhihu.com/p/52169807
    - https://zhuanlan.zhihu.com/p/52504407
    - https://zhuanlan.zhihu.com/p/61827629
- https://lumingdong.cn/engineering-practice-of-embedding-in-recommendation-scenario.html
- https://zhuanlan.zhihu.com/p/25343518
