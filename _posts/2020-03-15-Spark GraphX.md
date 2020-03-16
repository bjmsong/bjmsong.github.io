---
layout:     post
title:      Spark GraphX
subtitle:   
date:       2020-03-15
author:     bjmsong
header-img: img/spark/Spark_logo.png
catalog: true
tags:
    - Spark
---

- Overview
    - for graphs and graph-parallel computation
    - extends RDD by introducing a Graph abstraction
        - bssed on RDD
- 属性图 (Property Graph)
    - 有向多重图
      - 允许相同的顶点有多种关系：有多条平行的边
    - 顶点
      - VertexRDD
      - 顶点可以是不同类型：通过继承 
    - 边：EdgeRDD
    - 跟RDD一样： immutable, distributed, and fault-tolerant
    - triplets ： 三元组（结点-关系-结点）
- 图操作符 (Graph Operators)
    - Graph，GraphOps
        - scala  implicits 特性：GraphOps are automatically available as members of Graph
    - Property Operators
        - mapVertices，mapEdges，mapTriplets
            - 修改Vertices/Edges/Triplets之后，返回新的图
            - allows the resulting graph to reuse the structural indices of the original graph
    - Structural Operators
    - Join Operators
    - Neighborhood Aggregation
- cache，uncache
- **Pregel API**
  - For iterative computation we recommend using the Pregel API, which correctly unpersists intermediate results
  - https://blog.csdn.net/u013468917/article/details/51199808
    - 《Pregel: A System for Large-Scale Graph Processing》
- 构造图 (Builders)
- 图算法
    - PageRank
    - 连通图算法
    - 三角形计数算法：计算通过每个顶点的三角形的数量



### 参考资料

- [基于Spark GraphX实现微博二度关系推荐实践](https://www.weibo.com/ttarticle/p/show?id=2309404060500571876390)
- [graphx上的一些简单应用+代码](http://kubicode.me/2015/07/07/Spark/Graphs-Applications/)
- https://github.com/endymecy/spark-graphx-source-analysis