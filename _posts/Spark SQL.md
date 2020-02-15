---
layout:     post
title:      Spark SQL
subtitle:   
date:       2019-09-30
author:     bjmsong
header-img: img/spark/Spark_logo.png
catalog: true
tags:
    - Spark
---

Spark SQL是Spark的基本组件之一，主要用于结构化数据处理。可以通过sql和Dataset API和Spark SQL进行交互。

### SQL
可以执行sql语句直接和Hive交互，或者通过JDBC/ODBC和mysql、oracle等数据库交互。



### Datasets and DataFrames

Dataset是Spark 1.6引入的，具备RDD强类型、可以使用匿名函数等优点，同时又获得了Spark SQL的优化执行引擎。Dataset API目前只有Scala和Java，尚不支持Python。
DataFrame



### UDF

https://www.jianshu.com/p/b1e9d5cc6193
https://acadgild.com/blog/writing-a-custom-udf-in-spark



### Dataset,DataFrame,RDD

![rdd_dataframe_dataset]({{site.baseurl}}/img/spark/rdd_dataframe_dataset.png)

https://medium.zenika.com/a-comparison-between-rdd-dataframe-and-dataset-in-spark-from-a-developers-point-of-view-a539b5acf734
https://www.infoq.cn/article/three-apache-spark-apis-rdds-dataframes-and-datasets

- 相同：
	- 惰性机制
	- 根据内存情况自动缓存运算？
	- partition概念，如mappartition对每一个分区进行操作，数据量小，而且可以将运算结果拿出来，map中对外面变量的操作是无效的
	- 有共同的方法，如filter、排序等
- 不同：
    - RDD:不支持spark sql，支持spark mllib
    - DataFrame、Dataset：支持spark ml、spark sql
	- DataFrame、Dataset 比 RDD做了更多的优化 
- DataFrame VS Dataset
	- DataFrame：Dataset[Row]，每一行的类型是Row
	- Dataset访问列中的某个字段方便
- 互相转换
https://stackoverflow.com/questions/29383578/how-to-convert-rdd-object-to-dataframe-in-spark/42469625#42469625

```
DataFrame/Dataset 转RDD df.rdd
RDD转DataFrame：import spark.implicits._
				val testDF = rdd.map {line=>
						(line._1,line._2)
					}.toDF("col1","col2")
RDD转DataSet：import spark.implicits._
				case class Coltest(col1:String,col2:Int)extends Serializable //定义字段名和类型
				val testDS = rdd.map {line=>
						Coltest(line._1,line._2)0
					}.toDS
Dataset转DataFrame：   import spark.implicits._
						val testDF = testDS.toDF	
DataFrame转Dataset：	case class Coltest(col1:String,col2:Int)extends Serializable //定义字段名和类型
					val testDS = testDF.as[Coltest]		

printing elements of RDD：rddforeach(println)---print to the executor's stdout,not the driver's stdout;
                          rdd.collect().foreach(println)--print to the driver's stdout			   	    
```
- 优劣



#### 参考资料

https://spark.apache.org/docs/latest/sql-programming-guide.html