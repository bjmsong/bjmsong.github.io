---
layout:     post
title:      算法与数据结构之四
subtitle:   图
date:       2020-02-13
author:     bjmsong
header-img: img/Algo/algorithms.jpg
catalog: true
tags:
    - 算法
---
>图的主要内容是对象和它们的连接，连接可能有权重和方向。利用图可以为大量重要而困难的问题建模。本文将介绍深度优先搜索、广度优先搜索、连通性问题、最小树生成算法、最短路径算法等。

### Why study Algorithms

- Old root, new opportunities

- Their impact is broad and far-reaching

  - Internet,Biology,Pyhsics ...
  - Computational models are replacing math models in scientific inquiry

- Algorithms + Data Structures = Programs

- Bad programmers worry about code, good programmers worry about data structures and their relationships

- great algorithms are the poetry of computation

  ![](/home/redfish/Desktop/bjmsong.github.io/img/Algo/why study algorithms.png)



### 无向图

- 多重图、简单图
- 连通图
- 无环图
- 生成树
- 二分图
- 图的表示方法
    - 邻接表数组：以顶点为索引的列表数组，其中每个元素都是和该顶点相邻的顶点列表
    - 邻接集
- DFS 
- BFS
- 连通分量
    - dfs
- 符号图 
    - 顶点名为字符串



### 有向图

- 出度、入度
- 实现：邻接表
- 有向图的可达性
- 环和有向无环图(DAG)
    - 调度问题
        - 寻找有向环
        - 拓扑排序：给定一副有向图，将所有的顶点排序，使得所有的有向边均从排在前面的元素指向排在后面的元素
- 强连通性



### 最小生成树 MST

- 加权图
- 生成树：含有图的所有顶点的无环连通子图
- 最小生成树：权值之和最小的生成树



### 最短路径

- Dijkstra算法



### 参考资料

- Algorithms，Robert Sedgewick

  
