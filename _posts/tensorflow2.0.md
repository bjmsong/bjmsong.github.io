---
layout:     post
title:      TensorFLow 2.0
subtitle:   
date:       2019-09-24
author:     bjmsong
header-img: img/dl/tf.jpg
catalog: true
tags:
    - 深度学习
---
### 变化
- Eager Execution 变为 TensorFlow 2.0 默认的执行模式。这意味着 TensorFlow 如同 PyTorch 那样，由编写静态计算图全面转向了动态计算图。
- Eager 模式变为默认设置之后，开发者可以在原型搭建完成后，利用 AutoGraph 把在 Eager 模式下搭建的模型自动变成计算图。


### Google官方建议的最佳实践
- 用 Eager 模式搭建原型
- 用 Datasets 处理数据
- 用 Feature Columns 提取特征
- 用 Keras 搭建模型
- 借用 Canned Estimators
- 用 SavedModel 打包模型


### name_scope， variable_scope
https://blog.csdn.net/Jerr__y/article/details/70809528
- name_scope: 为了更好地管理变量的命名空间而提出的。比如在 tensorboard 中，因为引入了 name_scope， 我们的 Graph 看起来才井然有序。
- variable_scope: 大大大部分情况下，跟 tf.get_variable() 配合使用，实现变量共享的功能。

### tensorflow-serving


### 参考资料
- https://tensorflow.google.cn/beta
- https://mp.weixin.qq.com/s/EUICEJ-LTY28N6RTrp5HSg



- deeplearning.ai
- github热门教程
https://github.com/aymericdamien/TensorFlow-Examples
https://github.com/machinelearningmindset/TensorFlow-Course
https://github.com/pkmital/tensorflow_tutorials
- CS 20SI: Tensorflow for Deep Learning Research （b站）


https://github.com/tensorflow/models

- tensorflow 2.0 快速入门 (by Chollet) 
https://threader.app/thread/1105139360226140160
- deeplearning.ai 《Introduction to TensorFlow for AI,ML and DL》
- Udacity 《Intro to TensorFlow for Deep Learning》
- 《Tensorflow 实战google深度学习框架》
https://github.com/open-source-for-science/TensorFlow-Course

https://mp.weixin.qq.com/s?__biz=Mzg5ODAzMTkyMg==&mid=2247485526&idx=1&sn=1336d073b32606ce3e30ab493c550b6e&chksm=c069800bf71e091de2e0bd528d4bd511ce5c2f9ab262d3d9f61214abb3ce86e965c8337d5670&mpshare=1&scene=1&srcid=0402yRddQqeIPrG7ISjbxe8d&key=a4b793bdb8f1ac928bc12df678d75c7a1acbb0e5f4330732111c5ac55bd0b0f3e911aa814830e43fcd042f9e8d91df5f68b4185befa656b982b1205baeffedeb2df5c7e50e934cd0f2d1100d0aad872f&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+7&version=62060739&lang=zh_CN&pass_ticket=EbeRSFDHPEiyssEtBnc0InRbRW%2BHNu0QF5ryCTQpFOD6COzOY%2BH%2F9Q3Xqwl5u1cH
https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652042006&idx=4&sn=1ffa0bf4dc8c879150aeeea1756ef938&chksm=f12187e7c6560ef1be3cea7289f66ddcf7df55647f83a9c4963d91fa323e0c3276b9824328cd&mpshare=1&scene=1&srcid=&key=3ced8d6e9f21461a80ec1844ddf4bed938b54f25fd2729db4abf80f13e433df13e9df1b00410b0caf847b0fc30577ec020cbe01da7968db26b3f107104d5c80b2db25a727414dd5c9d9d6e07f31d325e&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+7&version=62060739&lang=zh_CN&pass_ticket=k%2BDwQUGMalJBoJr0NlWCUv0JX%2BaZPg%2B14DPHXkP7fNoOmY6S0Zm7FygiY2Gh97fp

- 常用模块代码
    https://github.com/taki0112/Tensorflow-Cookbook
    https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650757267&idx=4&sn=31cfed7163a6c049001b0a7a69a30eb2&chksm=871a9cedb06d15fbda2858747a4de2fee0cc8767beb019bc035ed71bd7a32fe1a43700a6bdf3&mpshare=1&scene=1&srcid=#rd
