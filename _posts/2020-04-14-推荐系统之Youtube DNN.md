---
layout:     post
title:      推荐系统之
subtitle:   Youtube DNN
date:       2020-04-14
author:     bjmsong
header-img: img/Recommendation System/th.jpg
catalog: true
tags:
    - 推荐系统
---



- 使用特征：用户观看过视频的embedding向量、用户搜索词的embedding向量、用户画像特征、context上下文特征等
- 训练方式：三层ReLU神经网络之后接softmax层，去预测用户下一个感兴趣的视频，输出是在所有候选视频集合上的概率分布。训练完成之后，最后一层Relu的输出作为user embedding，softmax的权重可当做当前预测item的embedding表示
- 线上预测：通过userId找到相应的user embedding，然后使用KNN方法（比如faiss）找到相似度最高的top-N条候选结果返回



### 参考资料
- Deep Neural Networks for YouTube Recommendations
- https://zhuanlan.zhihu.com/p/52169807
- https://zhuanlan.zhihu.com/p/52504407
- https://zhuanlan.zhihu.com/p/61827629
- https://zhuanlan.zhihu.com/p/25343518
- https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247486975&idx=1&sn=8c5bacd451b8d08a3517dc691872f6cf&chksm=fbd4bf93cca336858770741723368e687958d3f19ff20aec614d72e2ab17fdfe68a973d278a5&mpshare=1&scene=1&srcid=1209z8oqTqt1MpAnEKX0ob8h#rd