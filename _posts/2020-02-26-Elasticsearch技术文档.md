---

layout:     post
title:      Elasticsearch技术文档
subtitle:   
date:       2020-02-26
author:     bjmsong
header-img: img/cs/es/es.jpg
catalog: true
tags:
    - 信息检索
---



### Query DSL

- Elasticsearch提供基于JSON的完整Query DSL（域特定语言）来定义查询。可以将查询DSL看作查询的AST（抽象语法树），它由两类子句组成：

  - 叶查询子句：叶查询子句在特定字段中查找特定值，如 match，term或 range查询。这些查询可以自己使用

  - 复合查询子句：复合查询子句包装其他叶或复合查询，并用于以逻辑方式组合多个查询（如bool或dis_max查询），或更改它们的行为（如constant_score查询）

    

### Query & Filter

- Query 

  -  How well does this document match this query clause?
  -  calculates a `_score` representing how well the document matches, relative to other documents
  - 也可以实现精确匹配？
    - match_phase
  - 关键字：match，match_all，match_phrase

- Filter 

  - Does this document match this query clause

  - The answer is a simple Yes or No

  - **但是如果是字符串是text类型，直接filter往往匹配不上**

    - 因为text类型会先分词，然后根据分词后的内容建立反向索引，所以要通过分词后的词去filter

    - 查看字段类型

      ```
      GET /logstash-2020.03.10/_mapping
      ```

    - 查看分词结果

      ```
      GET /logstash-2020.03.10/_analyze
      {
        "field": "request_referer",
        "text": "roboams.datayes.com"
      }
      ```

    - keyword：不会分词，直接根据字符串内容建立反向索引

    - https://www.jianshu.com/p/1189ff372c38

    - https://www.jianshu.com/p/e1430282378d
  
  - 关键字：term，terms，range



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



### Scroll

- https://www.jianshu.com/p/14aa8b09c789?spm=a2c4e.11153940.blogcont326326.7.a8245169dIV2xx

- https://blog.csdn.net/u012089823/article/details/82258738

- https://blog.csdn.net/qq_32502511/article/details/93719442



### 参考资料

- [官方技术手册](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)

  

  

  



