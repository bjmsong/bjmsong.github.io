---
layout:     post
title:      Parameter Server
subtitle:   
date:       2022-11-11
author:     bjmsong
header-img: 
catalog: true
tags:
    - 并行计算
---
## 李沐讲 parameter server
https://www.bilibili.com/video/BV1YA4y197G8/?spm_id_from=333.999.0.0
- 领域规模大小：编程语言>>AI>>系统>>AI与系统交叉
- 挑战
    - 参数太多，网络开销太大
    - 机器学习都是顺序模型（一个批量算完再算下一个），大量的全局同步会影响性能
    - 容灾很重要：大规模任务失败概率很高（硬件、软件问题）
- 开源实现，适配各种机器学习算法
- 五个关键性特征
    - 通讯效率高：异步通讯，对算法进行压缩，降低通讯量
    - 灵活的一致性模型：各节点访问参数是否完全一致，牺牲一些算法精度
    - 弹性的可扩展性：训练的时候新的节点可以加进来，不会让整个任务停掉
    - 容灾：机器出问题时，多久可以恢复过来
    - 易用
- 实现
    - 服务节点：维护参数
        - 实时复制
    - 计算节点：请求服务节点参数，进行计算
        - 参数：稀疏向量、矩阵
##
《scaling distributed machine learning with the parameter server》
《Parameter Server for Distributed Machine Learning》
https://zhuanlan.zhihu.com/p/29968773
    - 专栏：分布式机器学习系统
https://www.zhihu.com/question/26998075
https://www.zhihu.com/question/53851014
参数服务器就类似于MapReduce，是大规模机器学习在不断使用过程中，抽象出来的框架之一。重点支持的就是参数的分布式，毕竟巨大的模型其实就是巨大的参数。
- 参数服务器是个编程框架，用于方便分布式并行程序的编写，其中重点是对大规模参数的分布式存储和协同的支持。
- 工业界需要训练大型的机器学习模型，一些广泛使用的特定的模型在规模上的两个特点：
1. 参数很大，超过单个机器的容纳能力（比如大型Logistic Regression和神经网络）
2. 训练数据巨大，需要分布式并行提速（大数据）


https://zhuanlan.zhihu.com/p/578046577?utm_id=0
https://www.cnblogs.com/rossiXYZ/p/16221579.html
