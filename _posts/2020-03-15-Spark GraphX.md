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

### Basic

- Overview
    - GraphX extends the Spark RDD by introducing a new Graph abstraction: a directed multigraph with properties attached to each vertex and edge. 
    - 支持快速构建图，以及在图上实现算法
- 属性图 (Property Graph)
    - 有向多重图：允许相同的顶点有多种关系，有多条平行的边
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
- 构造图 (Builders)
- 图算法
    - PageRank
    - 连通图算法
    - 三角形计数算法：计算通过每个顶点的三角形的数量



### Pregel

- For iterative computation we recommend using the Pregel API, which correctly unpersists intermediate results

- 在pregel中顶点有两种状态：活跃状态（active）和不活跃状态（halt）。如果某一个顶点接收到了消息并且需要执行计算那么它就会将自己设置为活跃状态。如果没有接收到消息或者接收到消息，但是发现自己不需要进行计算，那么就会将自己设置为不活跃状态。

- 计算过程：Pregel中的计算分为一个个“superstep”，这些”superstep”中执行流程如下：

  1. 首先输入图数据，并进行初始化。
2. 将每个节点均设置为活跃状态。每个节点根据预先定义好的sendmessage函数，以及方向（边的正向、反向或者双向）向周围的节点发送信息。
  3. 每个节点接收信息如果发现需要计算则根据预先定义好的计算函数对接收到的信息进行处理，这个过程可能会更新自己的信息。如果接收到消息但是不需要计算则将自己状态设置为不活跃。
  4. 每个活跃节点按照sendmessage函数向周围节点发送消息。
  5. 下一个superstep开始，像步骤3一样继续计算，直到所有节点都变成不活跃状态，整个计算过程结束。
  
- API

  ```scala
  // graphx.GraphOps 类
  def pregel[A](
      initialMsg: A, 
      maxIterations: Int = Int.MaxValue, 
      activeDirection: EdgeDirection = EdgeDirection.Either)(
      vprog: (VertexId, VD, A) ⇒ VD, 
      sendMsg: (EdgeTriplet[VD, ED]) ⇒ Iterator[(VertexId, A)], 
      mergeMsg: (A, A) ⇒ A)(implicit arg0: ClassTag[A]): Graph[VD, ED]
  ```

  - 采用的是典型的柯里化定义方式
  - 第一个括号中的参数序列分别为`initialMsg`、`maxIterations`、`activeDirection`
    - 第一个参数`initialMsg`表示第一次迭代时即superstep 0，每个节点接收到的消息
    - `maxIterations`表示迭代的最大次数，默认为Int.MaxValue
    - `activeDirection`表示消息发送的方向，该值为EdgeDirection类型，这是一个枚举类型，有三个可能值：EdgeDirection.In/ EdgeDirection.Out/ EdgeDirection.Either，默认为EdgeDirection.Either
  - 第二个括号中参数序列为三个函数，分别为`vprog`、`sendMsg`和`mergeMsg`
    - `vprog`是节点上的用户定义的计算函数，运行在单个节点之上，，在superstep 0，这个函数会在每个节点上以初始的`initialMsg`为参数运行并生成新的节点值。在随后的超步中只有当节点收到信息，该函数才会运行。
    - `sendMsg`在当前超步中收到信息的节点用于向相邻节点发送消息，这个消息用于下一个超步的计算
    - `mergeMsg`用于聚合发送到同一节点的消息，这个函数的参数为两个A类型的消息，返回值为一个A类型的消息。

- 例子：求单源最短路径

  ```scala
  import org.apache.spark._
  import org.apache.spark.graphx._
  import org.apache.spark.rdd.RDD
  
  val graph = GraphLoader.edgeListFile(sc,"/Spark/web-Google.txt") // 这个文件可以在https://snap.stanford.edu/data/web-Google.html下载
  val sourceId: VertexId = 0
  // 初始化图：对所有的非源顶点，将顶点的属性值设置为无穷
  // 因为我们打算将所有顶点的属性值用于保存源点到该点之间的最短路径,在正式开始计算之前将源点到自己的路径长度设为0，到其它点的路径长度设为无穷大，如果遇到更短的路径替换当前的长度即可。如果源点到该点不可达，那么路径长度自然为无穷大了
  val initialGraph = graph.mapVertices((id, _) => if (id == sourceId) 0.0 else Double.PositiveInfinity)
  
  // initialGraph会被隐式转换成GraphOps类
  val sssp = initialGraph.pregel(Double.PositiveInfinity)(
      (id, dist, newDist) => math.min(dist, newDist), // Vertex Program
      triplet => {                                    // Send Message
          if (triplet.srcAttr + triplet.attr < triplet.dstAttr) {
          Iterator((triplet.dstId, triplet.srcAttr + triplet.attr))
          } else {
          Iterator.empty
          }
      },
      (a,b) => math.min(a,b)                         // Merge Message
  )
  ```

  - 计算过程
    - Superstep 0：对所有顶点用initialmsg进行初始化，实际上这次初始化并没有改变什么
    - Superstep 1 : 对于每个triplet：计算`triplet.srcAttr+ triplet.attr` 和 `triplet.dstAttr`比较，以第一次为例：假设有一条边从0到a，这时就满足`triplet.srcAttr + triplet.attr < triplet.dstAttr`，这个`triplet.attr`的值实际上为1（没有自己指定，默认值都是1），而0的attr值我们早已初始化为0，0+1<无穷，所以发出的消息就是（a,1）这个在每个triplet中是从src发放dst的。如果某个边是从3到5，那么`triplet.srcAttr + triplet.attr < triplet.dstAttr`就不成立，因为无穷大加1等于无穷大，这时消息就是空的。
    - Superstep 1就是这样，这一步执行完后图中所有的与0直接相连的点的attr都成了1而且成为活跃节点，其它点的attr不变同时变成不活跃节点。活跃结点根据`triplet.srcAttr + triplet.attr < triplet.dstAttr`继续发消息，mergeMsg函数会对发送到同一节点的多个消息进行聚合，聚合的结果就是最小的那个值。
    - Superstep 2：所有收到消息的节点比较自己的attr和发过来的attr，将较小的值作为自己的attr。然后自己成为活节点继续向周围的节点发送attr+1这个消息，然后再聚合。
    - 直到没有节点的attr被更新，不再满足activeMessages > 0 && i < maxIterations （活跃节点数为大于0且没有达到最大允许迭代次数）。这时就得到节点0到其它节点的最短路径了。这个路径值保存在其它节点的attr中。



### 参考资料

- [基于Spark GraphX实现微博二度关系推荐实践](https://www.weibo.com/ttarticle/p/show?id=2309404060500571876390)

- [graphx上的一些简单应用+代码](http://kubicode.me/2015/07/07/Spark/Graphs-Applications/)

- https://github.com/endymecy/spark-graphx-source-analysis

- 《Pregel: A System for Large-Scale Graph Processing》

- https://blog.csdn.net/u013468917/article/details/51199808

  