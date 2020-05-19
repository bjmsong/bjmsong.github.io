---
layout:     post
title:      Algorithms之四
subtitle:   图
date:       2020-04-08
author:     bjmsong
header-img: img/Algo/algorithms.jpg
catalog: true
tags:
    - 算法
---
>图的主要内容是对象和它们的连接，连接可能有权重和方向。利用图可以为大量重要而困难的问题建模。本文将介绍深度优先搜索、广度优先搜索、连通性问题、最小树生成算法、最短路径算法等。

### Why study Algorithms

- Their impact is broad and far-reaching

  - Internet,Biology,Pyhsics ...
  - Computational models are replacing math models in scientific inquiry

- Algorithms + Data Structures = Programs

- Bad programmers worry about code, good programmers worry about data structures and their relationships

- great algorithms are the poetry of computation

  <ul> 
  <li markdown="1"> 
  ![]({{site.baseurl}}/img/Algo/why study algorithms.png) 
  </li> 
  </ul> 



### 无向图

- 术语
    - 多重图、简单图
    - 连通图
    - 无环图
    - 生成树：包含（连通）图中的所有顶点而且是一棵树
    - 二分图：能够将所有顶点分成两部分
    - 度
    - 稀疏、稠密
- 图的表示方法
    - 邻接矩阵：存储空间是O(V^2)，而且无法表示平行边
    - **邻接表数组**：以顶点为索引的列表数组，其中每个元素都是和该顶点相邻的顶点列表
    - 邻接集

```java
// API
public class Graph
    Graph(int V)   // 创建一个含有V个顶点，但不含边的图
    Graph(In in)   // 从标准输入流in读入一幅图
    int V()        // 顶点数
    int E()        // 边数
    void addEdge(int v int w)    // 添加边
    Iterable adj(inv v)   // 和v相邻的所有顶点
```
- 图搜索API
```java
publci class Serach
    Search(Graph G,int s)    // 找到和起点s连通的所有顶点
    boolean marked(int v)      // v和s是连通的吗
    int count()               // 和s连通的顶点总数
```
- 图搜索API的实现
    - union-find
    - DFS : 递归地访问所有的顶点，在访问其中一个顶点时：
    1. 将它标记为已访问
    2. 递归地访问所有它没有被标记过的邻居顶点
        - 无法解决最短路径的问题，因为DFS遍历图的顺序和找出最短路径这个目标没有任何关系
    - BFS：可以解决最短路径的问题
        - 要找到s到v的最短路径：从s开始，在所有由一条边就可以达到的顶点中寻找v，如果找不到，就在所有与s距离两条边的所有顶点中寻找v
- 找出一副图中所有的连通分量
    - DFS
    - 对比union-find：实际运行中，效率差不多（DFS搜索快，但是DFS需要构建图）
    - 只需要判断连通性，或者需要完成有大量连通性查询和插入操作混合 等类似任务：倾向于使用union-find算法
    - DFS更适合实现图的抽象数据类型
- DFS的其它应用
    - 检测环：给定的图是无环图吗
    - 双色问题：能够用两种颜色将图的所有顶点着色，任意一条边的两个端点的颜色都不相同
        - 等价问题：这是一个二分图吗
- 符号图 
  
    - 顶点名为字符串


### 有向图

- 术语：出度、入度
- 实现：邻接表
    - 用来表示有向图的数据结构甚至比无向图更简单：在无向图中一条表需要添加两次
- 有向图的可达性：与无向图的DFS代码完全相同

#### 环和有向无环图(DAG)
- 拓扑排序问题：给定一副有向图，将所有的顶点排序，使得所有的有向边均从排在前面的元素指向排在后面的元素
- 应用
    - 调度问题：给定一组任务和限制条件（如 优先级限制），安排它们的执行顺序
    - 课程安排
    - 继承
    - 电子表格
- 检测图是否是有向无环图 
    - 如果存在有向环，那肯定是无解的

#### 强连通性
- 顶点之间互相可达
- Kosaraju算法



### 最小生成树（MST）

- 加权无向图
- 生成树：含有图的所有顶点的无环连通子图
- 最小生成树：权值之和最小的生成树
- 树的性质
1. 用一条边连接树中的任意两个顶点都会产生一个环
2. 从树中删去一条边将得到两棵独立的树
- 加权无向图的表示
  - 加权边
  - 加权无向图

- Prim算法
  - 每一步都会为一棵生长中的树添加一条边
  - 每次总是将下一条连接树中的顶点与不在树中的顶点且权重最小的边加入树中
- Kruskal算法



### 最短路径

- 加权有向图

- 找到从一个顶点到达另一个顶点的成本最小的路径

- Dijkstra算法

  ```
  1. 初始化：源点的距离为0，其余顶点的距离为无穷大
  2. 先把源点加入最小堆
  3. 删除最小堆中距离最小的顶点v
  4. 遍历与v相连的顶点，松弛顶点，并更新最小堆
  5. 重复3-4步，直到最小堆为空
  ```

  - Dijkstra算法 vs Prime 算法
    - Prime 算法：每次添加离树最近的非树顶点
    - Dijkstra算法：每次添加离起点最近的非树顶点

- 无环加权有向图的最短路径算法

  - 比Dijkstra算法更快，更简单




### Maxflow/Mincut（最大流/最小割）

- 对于一个网络流，从源点到目标点的最大的流量等于最小割的每一条边的和
  - 即：图的最小割问题可以转换为最大流问题
- [https://imlogm.github.io/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/mincut-maxflow/](https://imlogm.github.io/图像处理/mincut-maxflow/)



### 参考资料

- Algorithms，Robert Sedgewick

  
