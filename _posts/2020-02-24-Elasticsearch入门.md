---

layout:     post
title:      Elasticsearch
subtitle:   
date:       2020-02-24
author:     bjmsong
header-img: img/cs/es/es.jpg
catalog: true
tags:
    - ES
---



> 全文搜索属于最常见的需求，开源的 Elasticsearch （以下简称 Elastic）是目前全文搜索引擎的首选。
>
> 它可以快速地储存、搜索和分析海量数据。维基百科、Stack Overflow、Github 都采用它。
>
> Elastic 的底层是开源库 Lucene。但是，你没法直接用 Lucene，必须自己写代码去调用它的接口。Elastic 是 Lucene 的封装，提供了 REST API 的操作接口，通过http请求就能对其进行操作。



### 基本概念

- Node 与 Cluster
  - Elastic 本质上是一个分布式数据库，允许多台服务器协同工作，每台服务器可以运行多个 Elastic 实例
  - 单个 Elastic 实例称为一个节点（node）。一组节点构成一个集群（cluster）

- Index
  - Elastic 会索引所有字段，经过处理后写入一个反向索引（Inverted Index）。查找数据的时候，直接查找该索引。
  - 所以，Elastic 数据管理的顶层单位就叫做 Index（索引）。它是单个**数据库**的同义词。每个 Index （即数据库）的名字必须是小写。

- Type 

  - 类型是用来定义数据结构的，**相当于mysql中的一张表**
  
  - Document 可以分组，比如weather这个 Index 里面，可以按城市分组（北京和上海），也可以按气候分组（晴天和雨天）。这种分组就叫做 Type，它是虚拟的逻辑分组，用来过滤 Document。
  
- Document 

  - Index 里面**单条的记录**称为 Document（文档）。许多条 Document 构成了一个 Index

  - Document 使用 JSON 格式表示，下面是一个例子

    ```
    {
      "user": "张三",
      "title": "工程师",
      "desc": "数据库管理"
    }
    ```

  - 同一个 Index 里面的 Document，不要求有相同的结构（scheme），但是最好保持相同，这样有利于提高搜索效率。
  
- field

  - 字段



#### 字符串类型

- keyword：不会分词，直接根据字符串内容建立反向索引

- text：先分词，然后根据分词后的内容建立反向索引



### Elasticsearch分布式原理

- master-slave架构
  - 在 Elasticsearch 中，节点是对等的，节点间会通过自己的一些规则选取集群的 Master，Master 会负责集群状态信息的改变，并同步给其他节点
- 会对数据进行切分，同时每一个分片会保存多个副本，其原因和 HDFS 是一样的，都是为了保证分布式环境下的高可用



### REST API

- 创建索引：索引名为poems

  ```
  curl -X PUT 'http://ip:port/poems'
  ```

- 删除索引

  ```
  curl -X DELETE 'http://ip:port/poems'
  ```

- 处理分词

  - 中文分词需要安装插件

  - 创建索引，指定需要分词的字段

    ```
    curl -X PUT 'localhost:9200/accounts' -d '
    {
      "mappings": {
        "person": {
          "properties": {
            "user": {
              "type": "text",
              "analyzer": "ik_max_word",
              "search_analyzer": "ik_max_word"
            },
            "title": {
              "type": "text",
              "analyzer": "ik_max_word",
              "search_analyzer": "ik_max_word"
            },
            "desc": {
              "type": "text",
              "analyzer": "ik_max_word",
              "search_analyzer": "ik_max_word"
            }
          }
        }
      }
    }'
    ```

    - 新建一个名称为accounts的 Index，里面有一个名称为person的 Type。person有三个字段：user、title、desc
    - 这三个字段都是中文，而且类型都是文本（text），所以需要指定中文分词器，不能使用默认的英文分词器。Elastic 的分词器称为 analyzer

- 新增/更新记录

  ```
  curl -X PUT 'localhost:9200/accounts/person/1' -d '
  {
    "user": "张三",
    "title": "工程师",
    "desc": "数据库管理"
  }' 
  ```

- 查看记录

  ```
  curl -X GET 'localhost:9200/accounts/person/1?pretty=true'
  ```

- 删除记录

  ```
  curl -X DELETE 'localhost:9200/accounts/person/1'
  ```

- 数据查询：直接请求/Index/Type/_search，就会返回所有记录

  ```
  curl -X GET 'localhost:9200/accounts/person/_search'
  ```

- 全文搜索：要求 GET 请求带有数据体

  ```
  curl -X GET 'localhost:9200/accounts/person/_search'  -d '
  {
    "query" : { "match" : { "desc" : "软件" }}
  }'
  ```

  - 上面代码使用 Match 查询，指定的匹配条件是desc字段里面包含"软件"这个词



### [Kibana](https://www.elastic.co/guide/cn/kibana/current/index.html)

- 开源的分析和可视化平台，设计用于和Elasticsearch一起工作
- 可以用Kibana来搜索，查看，并和存储在Elasticsearch索引中的数据进行交互
- 可以轻松地执行高级数据分析，并且以各种图表、表格和地图的形式可视化数据
- 插件：Dev Tools Console
- [Kibana](https://www.cnblogs.com/cjsblog/p/9476813.html)



#### DIscover

- 交互式探索数据
- 搜索数据语法
  - Kibana标准的查询语言（基于Lucene的查询语法）
  - 基于JSON的Elasticsearch查询语言DSL



#### Visualize

- 创建在你的Elasticsearch索引中的数据的可视化效果

  

#### Dashboard

- 显示可视化和搜索的集合

  

#### Monitoring

- 数据监控





### [Python Client -- elasticsearch](https://elasticsearch-py.readthedocs.io/en/master/#)

- Official low-level client for Elasticsearch
- For a more high level client library with more limited scope, have a look at elasticsearch-dsl
- ES中的高性能的部分大部分在helpers中实现
  - 如果要批量查询大量的数据，建议使用helpers.scan，helpers.scan返回的数据对象时迭代器，很大节省内存空间，而且查询速度要远远大于search；search在利用from、size参数控制返回数据的条数，scroll进行数据分页，也可以返回大数据，但是search返回的数据是以list的形式，如果一次需要返回的数据量比较大的话，则会十分耗费内存，而且数据传输速度也会比较慢
- [python操作es](https://www.jianshu.com/p/462007422e65)
- [getting-started-with-elasticsearch-in-python](https://towardsdatascience.com/getting-started-with-elasticsearch-in-python-c3598e718380)





### ELK系统

- Elasticsearch除了做搜索引擎，也可以搭建elk系统，也就是日志分析系统
- 其中 E 就是 Elasticsearch，L 是 Logstash，是一个日志收集系统，K 是 Kibana，是一个数据可视化平台

<ul> 
<li markdown="1"> 
![]({{site.baseurl}}/img/cs/es/elk.jpg) 
</li> 
</ul> 

- 是一整套日志（当然不仅限于日志）收集，转换，搜索，查询，可视化组件，对标于商业领域的Splunk。除了集中化的日志查询，它还能对日志内容进行索引，出各种漂亮的分析报表，是诊断问题，监控程序状态的一组非常有用的工具。
- [ELK](https://www.infoq.cn/article/architecture-practice-03-elk/)



### 参考资料

- [官方技术手册](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)

- [官方教程：Elasticsearch: 权威指南](https://www.elastic.co/guide/cn/elasticsearch/guide/cn/index.html)

- https://zhuanlan.zhihu.com/p/62892586

- http://www.ruanyifeng.com/blog/2017/08/elasticsearch.html

- https://www.cnblogs.com/cjsblog/p/9439331.html

- https://www.imooc.com/article/30131

  

  


