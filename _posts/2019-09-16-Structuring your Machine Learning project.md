---
layout:     post
title:      【笔记】deeplearning.ai之三
subtitle:   Structuring your Machine Learning project
date:       2019-09-16
author:     bjmsong
header-img: img/dl/dl.png
catalog: true
tags:
    - 深度学习
---


## week1.机器学习策略1
1.1 正交化 
按顺序调节以下目标，一个一个来，每个目标都有一组对应的方法
- 训练集上表现良好
    - bigger network，Adam。。
- 验证集上表现良好
    - Regularization，bigger training set
- 测试集上表现良好
    - bigger dev set
- 真实场景中表现良好
    - change dev set or cost function

“early stopping”不是很正交，会同时影响训练集和验证集的表现

1.2 单一数字评价指标
- 容易评估和比较
    - 比如：F1 score 比 precision&recall 更直观

1.3 优化指标&满足指标
- 优化指标：想要最大化的(比如F1)
- 满足指标：满足一定标准就可以了，不比追求最大化(比如运行时间)

1.4 change metrics and dev/test set
- 验证集、测试集：必须要来自同一个分布
- 训练集：验证集：测试集
    - 数据量非常大，验证集/测试集的比例可以降低 -- big enough to give high confidence in the overall performance of system
        - 0.98:0.01:0.01
- 如果评估指标无法正确挑选出好算法，那么就要修改评估指标
- 如果在dev/test 上表现良好，实际应用表现不佳
    - change metrics 
    - change dev/test set

1.5 优化步骤
- avoidable bias ：训练集误差和人类水平误差(近似认为是Bayes error：误差极限)之间的差距
- variance：验证集误差和训练集误差之间的差距
- 根据avoidable bias和variance的大小，确定优化的方向
    - 如果avoidable bias比较大：
        - train bigger model
        - train longer
        - better optimization algorithms
        - better NN architecture
        - hyperparameters search
    - 如果variance比较大：
        - mode data
        - regularization:L2,dropout
        - better NN architecture
        - hyperparameters search

## week2.机器学习策略2
2.1 误差分析：分析bad case
- 统计不同类型错误标记的例子的分布，分析错误的原因，优先解决影响最大的问题

2.2 训练集有错误标注怎么办
- DL algorithms are quite robust to **random errors** in the training set
    - 错误标记不要太多就行
- but not robust to **systemetic errors** 
    - 比如一直把白色的狗标记成猫
- 如果标记错误严重影响了在开发集上评估算法的能力，就应该花精力去修正标记错误

2.3 快速搭建系统，快速迭代
- 第一次处理某种问题，先实现完整的流程最重要
- set up train/dev/test set and metric
- build initial system quickly
- use bias/variance analysis and error analysis to prioritize next steps

2.4 训练集和验证集、测试集来自不同的分布
- 深度学习需要大量数据， 因此训练集数据越多越好，从各种渠道采集
    - 但是采集的样本和真实样本分布不一定一致，真实样本数量比较少
    - 这种情况下，不要将训练集和验证集混合，因为这样会降低验证集中真实样本的比例
    - option：验证集和测试集完全由真实样本组成，有多余的放入训练集
- bias & variance分析跟训练集/验证集来自同一个分布时的情况是不一样的
    - 需要将数据分成：train/train-dev/dev/test set
        - train-dev来自训练集，可以验证是否存在variance的问题
    - 比较以下error之间的差距：human level/training set error/training-dev set error/dev error/test error 
        --- 确定是以下哪种误差占主导：avoidable bias/variance/data mismatch/degree of overfiting to dev set
- data mismatch problem ： 了解训练集和测试集的差异，使训练集尽量接近测试集（比如人工加噪声）

2.5 迁移学习
- 从一个任务中学到的东西迁移到另外一个任务
- pre-train，fine-tunning
- step1：在大数据集上训练成熟网络
- step2：去除最后一层
- step3：在新数据集上重新训练部分层
- when transfer learning makes sense
    - Task A and B have the same input x
    - You have a lot more data for Task A than Task B
    - Low level features from A could be helpful for learning B

2.6 多任务学习(Multi-task learning)
- 目标变量y是多维的，最后一层也是多维的
- loss function(同时识别C个物体)
    $$L = 1/m\sum_i^m\sum_j^C L(y,y^-)$$ 
- 可以适用于只有部分标记的数据集
- when multi-task learning makes sense
    - Training on a set of tasks that could benefit from having shared lower-level features
    - Usually:Amount of data you have for each task is quite similar
    - Can train a big enough netural network to do well on all the tasks
- 物体检测任务

2.7 end-to-end deep learning
- 不需要数据处理/特征提取等中间流水线步骤
    - 用一个神经网络代替
    - 但是需要更多的数据
        - 如果数据没那么多，可以把问题拆成多步，这样每一步都有足够的数据，并且问题更简单
- Pros:
    - Let the data speak:让数据表现出自己的信息，而不是引入人类的成见
    - Less hand-designing of components needed
- Cons:
    - need large amount of data
    - Exclude potentially useful hand-designed components

