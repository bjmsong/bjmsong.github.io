---
layout:     post
title:      Build a faiss server
subtitle:   
date:       2022-12-06
author:     bjmsong
header-img: 
catalog: true
tags:
    - 信息检索
---
## brpc

## faiss
- 矢量检索：通过固定维度的浮点数矢量，在数据库里查找与之距离最近的 top-k 个矢量

https://faiss.ai/
https://github.com/facebookresearch/faiss
https://towardsdatascience.com/understanding-faiss-619bb6db2d1a
https://mp.weixin.qq.com/s?__biz=MjM5MzY4NzE3MA==&mid=2247485167&idx=1&sn=62489964ef77c48eab015c7bd9821cd6&chksm=a692750291e5fc14034a8b5bcdda9a1e626d1534569cc3c830f4d61675eaa03fb62650725024&mpshare=1&scene=1&srcid=&sharer_sharetime=1593565723594&sharer_shareid=52006a0d19edf83d2b8be98f4d8fe935&key=ffcaddc181c502c44aa70fdcdbd19e03b66046a0420fcd8d1dcef1bded2ac9e26bed37bccabd34018d648d50eb6f3654a72f43a104d5c722616a7bb2bfcaa436788ae9b5da3aa84e2d6bd3109b673c11&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+7+x64&version=62090523&lang=zh_CN&exportkey=AVCXRK4w2YDmfUz00Y5KE%2FE%3D&pass_ticket=631Kpf5GjrZ%2BLhtZjZK8xkjqZRzB9emP8BupzQIWz536OwxltUGB81K%2Fme%2FfC5Pm
https://zhuanlan.zhihu.com/p/40236865
https://www.cnblogs.com/paiandlu/p/12123859.html
https://www.gsitechnology.com/My-First-Adventures-in-Similarity-Search
https://blog.csdn.net/kanbuqinghuanyizhang/article/details/80774609

```python
import numpy as np

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32') # 训练数据
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32') # 查询数据
xq[:, 0] += np.arange(nq) / 1000.

# 创建索引,faiss创建索引对向量预处理，提高查询效率
index = basic_faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)

index.add(xb)                  # add vectors to the index
print(index.ntotal)

# 传入搜索向量查找相似向量
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(D[-5:])                  # neighbors of the 5 last queries
```

## build a faiss serving
https://www.jianshu.com/p/b9b422b3b119
https://www.jianshu.com/p/06cc695a8512
https://github.com/plippe/faiss-web-service
https://github.com/layerism/brpc_faiss_server
https://zhuanlan.zhihu.com/p/85510172
https://blog.csdn.net/zlb872551601/article/details/103704874
https://blog.csdn.net/xmxoxo/article/details/108884689
https://ailab-aida.github.io/2019/09/29/Ubuntu%20faiss%E5%AE%89%E8%A3%85%E5%B9%B6%E5%88%A9%E7%94%A8flask%E6%8F%90%E4%BE%9B%E5%90%91%E9%87%8F%E6%90%9C%E7%B4%A2%E6%9C%8D%E5%8A%A1API/