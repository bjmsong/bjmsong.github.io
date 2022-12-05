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
## https://huyenchip.com/2020/12/27/real-time-machine-learning.html
- Problems with batch predictions
    - This can work only when the input space is finite: 这样对用户体验会有很多限制
    - 不能获取用户最新的数据变化，预测的结果不是用户当前想要的
    - Twitter’s trending hashtag ranking, Facebook’s newsfeed ranking, estimating time of arrival
    - There are also many applications that, without online predictions, would lead to catastrophic failures or just wouldn’t work
        - high frequency trading, autonomous vehicles, voice assistants, unlocking your phones using face/fingerprints, fall detection for elderly care, fraud detection ...

1. Level 1: ML system makes predictions in real-time (online predictions)
- Fast inference: model that can make predictions in the order of milliseconds
    - Make models faster (inference optimization)
        - fusing operations
        - distributing computations
        - memory footprint optimization
        - writing high performance kernels targeting specific hardwares
    - Make models smaller (model compression)
        - quantization
        - knowledge distillation 
        - pruning 
    - Make hardware faster  
- Real-time pipeline: a pipeline that can process data, input it into model, and return a prediction in real-time
    - stores data as it stream: Apache Kafka
    - stream processing:Apache Flink
    - unify batch and stream processing pipelines with Flink
        - 传统情况下是batch一套(MapReduce/Spark/Hadoop),stream一套(Flink),两套不同架构会带来很多问题
        - https://open.163.com/newview/movie/free?pid=VFJGICSMA&mid=SFJGICSNB
    - Request-driven architecture works well for systems that rely more on logics than on data. Event-driven(消息队列) architecture works better for systems that are data-heavy
    - https://huyenchip.com/2022/08/03/stream-processing-for-data-scientists.html
    
2. Level 2: System can incorporate new data and update your model in real-time (continual learning)
- continual learning
    - stateful training: the model continues training on new data (fine-tuning)
    - 很多场景下需要
        - Recommendation systems for videos, articles, news, tweets, posts, memes
            - not needed because they are unlikely to change from a minute to the next: for houses, cars, flights, hotels
        - rare event
        - 实时热点
        - cold start
    - online evaluation
        - New models are first subject to offline tests to make sure they aren’t disastrous, then evaluated online in parallel with the existing models via a complex A/B testing system. Only when a model is shown to be better than an existing model in some metrics the company cares about that it can be deployed wider.
    - https://www.continualai.org/
    - https://www.youtube.com/watch?v=z9DDg2CJjeE&ab_channel=ContinualAI
    - https://github.com/ContinualAI
    - 《Continual lifelong learning with neural networks: A review》
    - https://github.com/GMvandeVen/continual-learning
    - https://towardsdatascience.com/how-to-apply-continual-learning-to-your-machine-learning-models-4754adcd7f7f


## https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html
### Online Prediction 
Stage 1. Batch prediction
- Batch prediction is NOT a prerequisite for online prediction. Batch prediction is largely a product of legacy systems.
- If you’re building a new ML system today, it’s possible to start with online prediction.
    - 现有的MapReduce，Spark系统是用来处理数据的

Stage 2. Online prediction with batch features
- 以session-based recommendation为例
    - 根据用户访问的item，得到item的向量，做平均得到用户向量
    - 召回
        - item-item协同过滤
        - 最近邻检索
    - 排序
- Requirements
    - Update your models from batch prediction to session-based predictions.
    - Integrate session data into your prediction service.
        - a streaming transport
        - a streaming computation engine
- Challenges
    - Inference latency
    - Setting up the streaming infrastructure
    - Having high-quality embeddings

Stage 3. Online prediction with complex streaming + batch features
- Requirements
    - Mature streaming infrastructure with an efficient stream processing engine that can compute all the streaming features with acceptable latency
    - A feature store for managing materialized features and ensuring consistency of stream features during training and prediction
    - A model store
    - Preferably a better development environment

#### Online prediction for bandits and contextual bandits
- Online prediction not only allows your models to make more accurate predictions but also enables bandits for online model evaluation – which is more interesting and more powerful than A/B testing
    - When you have multiple models to evaluate, each model can be considered a slot machine whose payout (e.g. prediction accuracy) you don’t know. Bandits allow you to determine how to route traffic to each model for prediction to determine the best model while minimizing wrong predictions shown to your users
- Bandits require less data to determine which model is the best, and at the same time, reduce opportunity cost as they route traffic to the better model more quickly
- https://towardsdatascience.com/a-b-testing-is-there-a-better-way-an-exploration-of-multi-armed-bandits-98ca927b357d
- https://blog.csdn.net/sinat_26917383/article/details/125048193
- Requirements
    - Online prediction
    - Preferably short feedback loops
    - A mechanism to collect feedback, calculate and keep track of each model’s performance, as well as route prediction requests to different models based on their current performance
- Contextual bandits
    - If bandits for model evaluation are to determine the payout (e.g. prediction accuracy) of each model, contextual bandits are to determine the payout of each action

### Continual Learning
- Stage 1. Manual, Stateless Retraining
- Stage 2. Automated Retraining
    - Requirements
        - Scheduler: Airflow
        - Data access and availability
    - Bonus: Log and wait (feature reuse)
        - reuse these extracted features for model updates, which both saves computation and allows for consistency between prediction and training
- Stage 3. Automated, Stateful Training
    - Stateful retraining allows you to update your model with less data
    - Requirements
        - Model lineage: you want to not just version your models but also track their lineage – which model fine-tunes on which model.
        - Streaming features reproducibility: you want to be able to time-travel to extract streaming features in the past and recreate the training data for your models at any point in the past in case something happens and you need to debug.
- Stage 4. Continual Learning
    - Instead of updating your models based on a fixed schedule, continually update your model whenever data distributions shift and the model’s performance plummets.
    - Requirements
        - A mechanism to trigger model updates
        - Better ways to continually evaluate your models.
        - An orchestrator

## https://eugeneyan.com/writing/real-time-recommendations/
- When does real-time recommendation make sense? When does it not?
    - Relative to real-time recommendations, batch recommendations are computationally cheaper
    - Batch recommendations are also simpler ops-wise
    - real-time recommendations are useful when the customer journey is mission-centric and depends on the context. 
- How have China and US companies implemented real-time recommenders?
- How can we design and implement a simple MVP?

## FTRL:Follow The Regularized Leader
- SGD算法是常用的online learning算法，它能学习出不错的模型，但学出的模型不是稀疏的。
- http://vividfree.github.io/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2015/12/05/understanding-FTRL-algorithm
https://www.cnblogs.com/EE-NovRain/p/3810737.html
https://www.cnblogs.com/massquantity/p/12693314.html
- Google 《Ad Click Prediction:a View from the Trenches》 
    - https://blog.csdn.net/u011239443/article/details/80528717
- https://github.com/guocheng2018/FTRL-pytorch
- https://github.com/fmfn/FTRLp
- https://github.com/CNevd/Difacto_DMLC
- 腾讯Angel https://github.com/Angel-ML/angel/blob/master/docs/algo/ftrl_lr_spark.md



## 参考资料
- https://blog.csdn.net/yz930618/article/details/75270869
- https://www.kaggle.com/titericz/giba-darragh-ftrl-rerevisited
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