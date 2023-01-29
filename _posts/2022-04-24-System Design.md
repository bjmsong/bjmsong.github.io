---
layout:     post
title:      System Design
subtitle:   
date:       2022-04-24
author:     bjmsong
header-img: 
catalog: true
tags:
    - 
---
## 花花酱
- https://www.youtube.com/watch?v=PMCdWr6ejpw&list=PLLuMmzMTgVK4RuSJjXUxjeUt3-vSyA1Or&ab_channel=HuaHua
### Design Twitter
- Clarify the requirements
    - Functional Requirement
         - Create Tweet
         - Read Timeline/Feed: Home, User
         - ...
    - Non-Functional Requirement
        - Consistency 
            - Every read receives the most recent write or an error
        - Availability 可用性
            - Every request receives a non-error response, without the guarantee that it contains the most recent write
            - Scalable
        - Fault tolerance
- Capacity Estimation
    - 存储
    - 带宽
- Design system APIs
- High-level System Design
    - load balancer
    - cache
        - policy：LRU，LFU
    - fan out on wirte
- Data Storage
    - DB表设计
    - SQL database，NoSQL database、File system
- Scalability
    - Data sharding（分区）
    - Load balancing
    - Data caching
        - useful for read heavy system

### Design Youtube/Bilibili
- High-level System Design
    - Message Queue：异步处理
    - CDN：用户可以就近获取内容，适合静态内容（不会改动），存放热门内容
- 

## 《System Design Interview: An Insider’s Guide》
- web server, DNS，load balancer，CDN，cache, stateless web tier，Message queue，微服务
- database：relational/non-relational, replication（master-slaver）,shard
- Logging, metrics, automation（如持续集成）
- 面试技巧：Clarify requirements（沟通设计目标），Capacity estimation（估算存储/qps/带宽等），system API，high-level system Design，Data storage，Scalability(shard,cache,load balance)

## 参考资料
- **https://github.com/donnemartin/system-design-primer** 
- https://schelleyyuki.com/learn-system-design
- **Grokking the System Design Interview**
    - https://www.youtube.com/channel/UCRPMAqdtSgd0Ipeef7iFsKw   
    - https://www.youtube.com/channel/UCn1XnDWhsLS5URXTi5wtFTA
    - https://www.youtube.com/channel/UCZLJf_R2sWyUtXSKiKlyvAw 
- https://www.1point3acres.com/bbs/thread-683982-1-1.html
- https://www.freecodecamp.org/news/a-thorough-introduction-to-distributed-systems-3b91562c9b3c/
- youtube频道：Tech Dummies，System Design Interview，InfoQ，@Scale，Data council
    - https://www.youtube.com/watch?v=BkSdD5VtyRM
- https://www.1point3acres.com/bbs/thread-605509-1-1.html
    - 极客时间：即时消息技术剖析与实战  
- https://medium.com/system-design-blog
https://www.1point3acres.com/bbs/forum-323-1.html
https://www.1point3acres.com/bbs/thread-169343-1-1.html
https://www.1point3acres.com/bbs/thread-456546-1-1.html
https://www.1point3acres.com/bbs/thread-545190-1-1.html
https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=456546&extra=
https://www.1point3acres.com/bbs/thread-552608-1-1.html
http://www.mitbbs.com/article_t/JobHunting/32777529.html
https://www.freecodecamp.org/news/a-thorough-introduction-to-distributed-systems-3b91562c9b3c/
- 《I heart logs》
- 知乎：阿莱克西斯
- 《Software Architecture for Developers》
- http://www.designsmells.com/articles/ten-must-to-read-books-for-software-architects/
- 端到端一致性,流系统Spark/Flink/Kafka/DataFlow对比总结(压箱宝具呕血之作)
https://zhuanlan.zhihu.com/p/77677075?utm_source=wechat_session&utm_medium=social&s_r=0#showWechatShareTip
- fedds 系统设计
https://mp.weixin.qq.com/s?__biz=MzIyNjE4NjI2Nw==&mid=2652561856&idx=1&sn=3656879af1cea79041b75bfe7a231572&chksm=f39a04b4c4ed8da26be8b73973861d73811cb2e4a0b0d18a5ce7566347b10695af69f74f1365&mpshare=1&scene=1&srcid=09163Vpl7DiVry3K00igVSoR&sharer_sharetime=1568675662116&sharer_shareid=49581f7bdbef8664715f595bc62d7044&key=c86a338f58bd007c3daf27f7d616fe005a4e374d136496fbff2b1c27a31d26ed5ebcea5c5288a4893c6beccf6f1ddf561637cc5f606623c01212752743a66d96b2fae81346acc95ed5e7ee80cabd6b4f&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+7&version=62060833&lang=zh_CN&pass_ticket=fZx3R6hAbiq5JXQFJM5ZrEYTbNGY4PKgyOa4uD91VEjM4%2FQUCMtw94n3V1bJeCDB
https://zhuanlan.zhihu.com/p/76541337?utm_source=wechat_session&utm_medium=social&s_s_i=fg41NsREUSsyBOQAo/lk+HPO5qCB2tw7PkZHF1Yx0cA=&s_r=1


