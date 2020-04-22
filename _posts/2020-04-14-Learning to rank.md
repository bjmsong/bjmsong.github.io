---
layout:     post
title:      Learning to Rank
subtitle:   
date:       2020-04-14
author:     bjmsong
header-img: img/IR/ir.jpg
catalog: true
tags:
    - 信息检索
---



### Introduction

- Many IR problems are by nature ranking problems
  - document retrieval, collaborative filtering, key term extraction, definition finding, important email routing, sentiment analysis, product rating, and anti Web spam
- Learning-to-rank: learn how to combine predefined features for ranking by means of discriminative learning



#### Ranking in IR

- Conventional Ranking Models for IR

  - Query-dependent models
    - Boolean model 
      - based on the occurrences of the query terms in the documents
      - can only predict whether a document is relevant to the query or not, but cannot predict the degree of relevance
    - Vector Space model(VSM)
      - documents and queries are represented as vectors in a Euclidean space
        - TF-IDF
      - similarities: the inner product of two vectors 
    - Latent Semantic Indexing(LSI)
      - 
  - Query-independent models
    - PageRank

- Query-level Position-based Evaluations in IR

  

#### Learning to Rank

- ML Framework
- Learning-to-rank Framework
- Approaches to Learning to Rank



#### About this Tutorial





### The Pointwise Approach



### The Pairwise Approach



### The Listwise Approach



###  Analysis of the Approaches




### 列表评价指标
- 挑战 
    - 定义文章与关键词之间的相关度，这决定了一篇文章在列表中的位置，相关度越高排序就应该越靠前
    - 当列表中某些文章没有排在正确位置的时候，如何给整个列表打分。
- 三个阶段
    - Precision & Recall
        - 对于一个请求关键词，所有文档被标记为相关和不相关两种
        - 举个列子来说，对于某个请求关键词，有200篇文章实际相关。某个排序算法只认为100篇文章是相关的，而这100篇文章里面，真正相关的文章只有80篇。按照以上定义：
            准确率=80/100=0.8
            召回率=80/200=0.4
        - 缺点：没有考虑位置因素
    - Discounted Cumulative Gain(DCG)
        - 对于一个关键词，所有的文档可以分为多个相关性级别，这里以rel1，rel2...来表示。文章相关性对整个列表评价指标的贡献随着位置的增加而对数衰减，位置越靠后，衰减越严重。基于DCG评价指标，列表前p个文档的评价指标定义如下：
            $ DCG_p=\sum_{i=1}^{p}\frac{2^{rel_i}-1}{log_2(i+1)} $
        - 因为不同请求的结果列表长度往往不同，工业界常用的是Normalized DCG(nDCG),假定能够获取到某个请求的前p个位置的完美排序列表，这个完美列表的分值称为Ideal DCG(IDCG)，nDCG等于DCG与IDCG比值。所以nDCG是一个在0到1之间的值。
            $ nDCG_p = \frac{DCG_p}{IDCG_p}$
    - Excepted Reciprocal Rank(ERR)
        - 与DCG相比，除了考虑位置衰减和允许多种相关级别（以R1，R2，R3...来表示）以外，ERR更进了一步，还考虑了排在文档之前所有文档的相关性。
        - 举个例子来说，文档A非常相关，排在第5位。如果排在前面的4个文档相关度都不高，那么文档A对列表的贡献就很大。反过来，如果前面4个文档相关度很大，已经完全解决了用户的搜索需求，用户根本就不会点击第5个位置的文档，那么文档A对列表的贡献就不大。
        $ ERR = \sum_{r=1}^n\frac{1}{r}\prod_{i=1}^{r-1}(1-R _i)R_r$

### 列表训练算法
以给学生排名为例
- Pointwise
    - 已知每个学生的成绩
    - 例如：CTR预估
    - 最容易做
- Pairwise
    - 已知任意两个学生的互相排名
    - Lambda系列：LambdaRank，LambdaMart
    - 核心思想是：很多时候我们很难直接计算损失函数的值，但却很容易计算损失函数梯度（Gradient）。这意味着我们很难计算整个列表的nDCG和ERR等指标，但却很容易知道某个文档应该排的更靠前还是靠后。
- Listwise
    - 已知所有学生的整体排名
    - 效果往往最好，但是如何为每个请求对所有文档进行标注是一个巨大的挑战。


### 参考资料

- 《Learning to Rank for Information Retrieval》Tie-Yan Liu

