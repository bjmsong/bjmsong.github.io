---
layout:     post
title:      Naive Bayes
subtitle:   
date:       2019-11-30
author:     bjmsong
header-img: img/machineLearning/machineLearning.png
catalog: true
tags:
    - 机器学习
---
>朴素贝叶斯（Naive Bayes）是基于贝叶斯定理与特征条件独立假设的分类方法，属于生成模型。



### 算法

1. 计算先验概率及条件概率（极大似然估计）

   $$P(Y=c_k)$$

   $$P(X^{(j)}=a_{jl}|Y=c_k)$$

   ​	**朴素贝叶斯对条件概率作了条件独立性的假设，牺牲了准确性**

2. 对于给定的实例x，计算确定x的类

   $$y = argmax_{c_k} P(Y=c_k) \prod_{j=1}^nP(X^{j} = x^{j} | Y=c_k)$$

- 概率值为0的情况，进行拉普拉斯平滑
- 代码实现
