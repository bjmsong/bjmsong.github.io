---
layout:     post
title:      TensorFLow
subtitle:   
date:       2019-09-16
author:     bjmsong
header-img: img/dl/tf.jpg
catalog: true
tags:
    - 深度学习
---


# 2.0即将发布，跟1.0相比，使用方法有重大变动

## 介绍
TensorFlow™ 是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。
- 使用图(graph)来表示计算任务
    - 节点称之为op(operator),一个op获得0个或多个Tensor，执行计算，返回0个或多个Tensor
- 在被称之为会话(Session)的上下文(context)中执行图
    - 目的是节省资源开销？
    - 图必须在会话里被启动
    - 会话将图的op分发到CPU、GPU
    - 一般不需要显式指定CPU/GPU,TensorFLow会自动检测，优先使用GPU
    - 如果机器上有超过一个可用的 GPU, 除第一个外的其它 GPU 默认是不参与计算的. 为了让 TensorFlow 使用这些 GPU, 你必须将 op 明确指派给它们执行
- 使用tensor表示数据
    - 类型化的多维数组：np.ndarray
- 通过变量(Variable)维护状态
- 使用feed和fetch可以为任意的操作赋值或者从其中获取数据

- 通常分成两个阶段
    - 构建阶段：将op的执行步骤，描述成一个图
    - 执行阶段：在会话中执行图中的op

## name_scope， variable_scope
https://blog.csdn.net/Jerr__y/article/details/70809528
- name_scope: 为了更好地管理变量的命名空间而提出的。比如在 tensorboard 中，因为引入了 name_scope， 我们的 Graph 看起来才井然有序。
- variable_scope: 大大大部分情况下，跟 tf.get_variable() 配合使用，实现变量共享的功能。

## 参考资料
https://github.com/aymericdamien/TensorFlow-Examples
- https://github.com/machinelearningmindset/TensorFlow-Course
- tensorflow 2.0 快速入门 (by Chollet) 
https://threader.app/thread/1105139360226140160
- deeplearning.ai 《Introduction to TensorFlow for AI,ML and DL》
- Udacity 《Intro to TensorFlow for Deep Learning》
- 《Tensorflow 实战google深度学习框架》
http://www.tensorflow.org/

https://www.tensorflow.org/tutorials/
https://github.com/open-source-for-science/TensorFlow-Course

https://mp.weixin.qq.com/s?__biz=Mzg5ODAzMTkyMg==&mid=2247485526&idx=1&sn=1336d073b32606ce3e30ab493c550b6e&chksm=c069800bf71e091de2e0bd528d4bd511ce5c2f9ab262d3d9f61214abb3ce86e965c8337d5670&mpshare=1&scene=1&srcid=0402yRddQqeIPrG7ISjbxe8d&key=a4b793bdb8f1ac928bc12df678d75c7a1acbb0e5f4330732111c5ac55bd0b0f3e911aa814830e43fcd042f9e8d91df5f68b4185befa656b982b1205baeffedeb2df5c7e50e934cd0f2d1100d0aad872f&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+7&version=62060739&lang=zh_CN&pass_ticket=EbeRSFDHPEiyssEtBnc0InRbRW%2BHNu0QF5ryCTQpFOD6COzOY%2BH%2F9Q3Xqwl5u1cH
https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652042006&idx=4&sn=1ffa0bf4dc8c879150aeeea1756ef938&chksm=f12187e7c6560ef1be3cea7289f66ddcf7df55647f83a9c4963d91fa323e0c3276b9824328cd&mpshare=1&scene=1&srcid=&key=3ced8d6e9f21461a80ec1844ddf4bed938b54f25fd2729db4abf80f13e433df13e9df1b00410b0caf847b0fc30577ec020cbe01da7968db26b3f107104d5c80b2db25a727414dd5c9d9d6e07f31d325e&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+7&version=62060739&lang=zh_CN&pass_ticket=k%2BDwQUGMalJBoJr0NlWCUv0JX%2BaZPg%2B14DPHXkP7fNoOmY6S0Zm7FygiY2Gh97fp

- 常用模块代码
    https://github.com/taki0112/Tensorflow-Cookbook
    https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650757267&idx=4&sn=31cfed7163a6c049001b0a7a69a30eb2&chksm=871a9cedb06d15fbda2858747a4de2fee0cc8767beb019bc035ed71bd7a32fe1a43700a6bdf3&mpshare=1&scene=1&srcid=#rd
tenserboard
