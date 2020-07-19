---
layout:     post
title:      Online Learning
subtitle:   
date:       2020-04-22
author:     bjmsong
header-img: img/machineLearning/machineLearning.png
catalog: true
tags:
    - 机器学习
---
### 定义
- 传统的学习算法是batch learning算法，它无法有效地处理大规模的数据集，也无法有效地处理大规模的在线数据流。这时，有效且高效的online learning算法显得尤为重要。
- SGD算法是常用的online learning算法，它能学习出不错的模型，但学出的模型不是稀疏的。



### 算法

- SGD
- FTRL（Follow The Regularized Leader）
http://vividfree.github.io/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2015/12/05/understanding-FTRL-algorithm
https://www.cnblogs.com/EE-NovRain/p/3810737.html



### FTRL工程实现

- Google 《Ad Click Prediction:a View from the Trenches》 
- 腾讯Angel https://github.com/Angel-ML/angel/blob/master/docs/algo/ftrl_lr_spark.md



### 参考资料

- 《在线最优化求解》 有pdf
- Adaptive Bound Optimization for Online Convex Optimization
- https://zhuanlan.zhihu.com/p/36410780
- https://www.zhihu.com/question/37426733
- https://zhuanlan.zhihu.com/p/77664408

https://en.wikipedia.org/wiki/Online_machine_learning
https://courses.cs.washington.edu/courses/cse599s/12sp/index.html
https://mlwave.com/online-learning-perceptron/
https://blog.csdn.net/yz930618/article/details/75270869
https://blog.csdn.net/dengxing1234/article/details/73277251
https://medium.com/value-stream-design/online-machine-learning-515556ff72c5
https://www.quora.com/What-is-the-best-way-to-learn-online-machine-learning
https://github.com/creme-ml/creme
https://daiwk.github.io/posts/ml-ftrl.html