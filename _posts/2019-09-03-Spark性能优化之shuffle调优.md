---
layout:     post
title:      【转载】Spark性能优化之四
subtitle:   shuffle调优
date:       2019-09-03
author:     bjmsong
header-img: img/spark/Spark_logo.png
catalog: true
tags:
    - Spark
---


#### **尽量使用map类的非shuffle算子**

详见《Spark性能优化之数据倾斜调优》

#### 使用map-side预聚合的shuffle操作

所谓的map-side预聚合，说的是在每个节点本地对相同的key进行一次聚合操作。map-side预聚合之后，每个节点本地就只会有一条相同的key，其他节点在拉取所有节点上的相同key时，就会大大减少需要拉取的数据数量，从而也就减少了磁盘IO以及网络传输开销。

通常来说，在可能的情况下， **建议使用reduceByKey或者aggregateByKey算子来替代掉groupByKey算子。** 因为reduceByKey和aggregateByKey算子都会使用用户自定义的函数对每个节点本地的相同key进行预聚合。而groupByKey算子是不会进行预聚合的，全量的数据会在集群的各个节点之间分发和传输，性能相对来说比较差。

#### mapValue 比 map 好 ?

明确 key 不会变的 map，就用 mapValues 来替代，因为这样可以保证 Spark 不会 shuffle 你的数据

#### 明确哪些操作必须在 master 完成

如果想打印一些东西到 stdout 里去：
```
A.foreach(println)
```
想把 RDD 的内容逐条打印出来，但是结果却没有出现在 stdout 里面，因为这一步操作被放到 slave 上面去执行了。其实只需要 collect 一下，这些内容就被加载到 master 的内存中打印了：
```
A.collect.foreach(println)
```
**再比如，如果RDD 操作嵌套的情况是不支持的**，因为只有 master 才能去理解和执行 RDD 的操作，slave 只能处理被分配的 task 而已。比如：
```
A.map{case (keyA, valueA) => doSomething(B.lookup(keyA).head, valueA)}
```
就可以用 join 来代替：
```
A.join(B).map{case (key, (valueA, valueB)) => doSomething(valueB, valueA)}
```

#### reaprtition(窄依赖，避免shuffle)： 
hash-partitioned, rdd.partitionBy(new HashPartitioner(num)).persist()
dataframe.repartition(num，colname to join) -- The resulting Dataset is hash partitioned. This is the same operation as   "DISTRIBUTE BY" in SQL (Hive QL). 

#### 向量化、并行化的代码写法
#### 不要重复造轮子
[SQL built-in functions](http://spark.apache.org/docs/latest/api/sql/index.html).
#### 调整GC
- RDD很多时GC会成为一个瓶颈
- 首先可以统计程序在GC上花的时间
- GC上花的时间跟Java对象的数量成正比
- 对象越少越好
- 序列化数据

### 参考资料
- https://www.raychase.net/3788
- https://blog.csdn.net/fl63zv9zou86950w/article/details/79049280
