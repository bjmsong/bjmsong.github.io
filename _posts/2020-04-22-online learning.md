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
### continual learning
- https://towardsdatascience.com/how-to-apply-continual-learning-to-your-machine-learning-models-4754adcd7f7f
- https://zhuanlan.zhihu.com/p/82540025

### Online Learning定义
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

- https://mp.weixin.qq.com/s?__biz=MjM5MzY4NzE3MA==&mid=2247485716&idx=1&sn=106f5d6b17294260d7259e2d44ba8f07&chksm=a6927af991e5f3ef83b80a7d13f31029bd8fafd648f11ec6060a7b512089ef9b2a2b1086dbaa&mpshare=1&scene=1&srcid=0721pInQ6zlzDBRqZHG4Y7hD&sharer_sharetime=1595292571063&sharer_shareid=52006a0d19edf83d2b8be98f4d8fe935&key=25b7ee6511d12c93dda7ff22600e8b92d169a2acf5bbea0eb0203bae0b8688448669e519aaa07a8f7c207d52a8f04beeb914a29178ecd370024146e039d5c6d3ce865e2e3454144ee4e97932fdb4c700&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+7+x64&version=62090529&lang=zh_CN&exportkey=AcFfhjRSc6ctODLTCYjIHKY%3D&pass_ticket=ipbSwC99tbDlmwwBuZrvYZcIonVi64LqRihIOgOYXl%2BzSFTLEDMbBZ6xvOTlh6Kn