---

layout:     post
title:      Elasticsearch技术文档
subtitle:   
date:       2020-02-26
author:     bjmsong
header-img: img/cs/es/es.jpg
catalog: true
tags:
    - ES
---





### 聚合

- 四种聚合类型

  - Bucketing（桶聚合）

    - 返回很多子集，并限定输入数据到一个特殊的叫做桶的子集中。可以把桶聚合想象成

      类似前面切面功能的东西。

  - Metric（度量聚合）

    - 度量聚合接收一个输入文档集并生成至少一个统计值sum
    - sum
    - count：count(*)
    - stats:  返回所有度量类型的统计
    - avg
    - min,max

  - Matrix

  - Pipeline



https://zhuanlan.zhihu.com/p/37500880



### Query DSL

- Elasticsearch提供基于JSON的完整Query DSL（域特定语言）来定义查询。可以将查询DSL看作查询的AST（抽象语法树），它由两类子句组成：

  - 叶查询子句：叶查询子句在特定字段中查找特定值，如 match，term或 range查询。这些查询可以自己使用
  - 复合查询子句：复合查询子句包装其他叶或复合查询，并用于以逻辑方式组合多个查询（如bool或dis_max查询），或更改它们的行为（如constant_score查询）

#### Query and filter context

- The behaviour of a query clause depends on whether it is used in query context or in filter context
- Query context
  - 
- Filter context





### 参考资料

- [官方技术手册](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)

  

  

  



