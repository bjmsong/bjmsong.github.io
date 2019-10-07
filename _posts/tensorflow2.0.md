---
layout:     post
title:      TensorFLow 2.0
subtitle:   
date:       2019-09-24
author:     bjmsong
header-img: img/dl/tf2.0.png
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



https://mp.weixin.qq.com/s?__biz=MzIzNjc1NzUzMw==&mid=2247530867&idx=1&sn=e0855bd03cbedde94df8cf947b9d8a68&chksm=e8d0ce01dfa74717ab43f773af0c5add8c8b7869ab13549f7f3eae6200faf1c2d739b7ac398c&mpshare=1&scene=1&srcid=&sharer_sharetime=1569996302466&sharer_shareid=49581f7bdbef8664715f595bc62d7044&key=8a4bbb55c6c79ce6fa454f5f47b52efb672dd9d3c51b557d7e3f94b262546f8efda62a6b5db95a39bdf13e98095ff283349bebd5031578f6e674caa30dc82acb1bc085b293576cb51894c84e64f645f0&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+10&version=62070141&lang=en&pass_ticket=doB3BRar8EovetMHBhmSUx44fADeRFqbXn4STwtBWkUO4U8Pw6IebWbKuRSVBN6v


### name_scope， variable_scope
https://blog.csdn.net/Jerr__y/article/details/70809528
- name_scope: 为了更好地管理变量的命名空间而提出的。比如在 tensorboard 中，因为引入了 name_scope， 我们的 Graph 看起来才井然有序。
- variable_scope: 大大大部分情况下，跟 tf.get_variable() 配合使用，实现变量共享的功能。

### tensorflow-serving
https://blog.csdn.net/shin627077/article/details/78592729
https://blog.csdn.net/heyc861221/article/details/80129169
https://www.jianshu.com/p/2fffd0e332bc

### 参考资料
- https://tensorflow.google.cn/beta
- https://mp.weixin.qq.com/s/EUICEJ-LTY28N6RTrp5HSg



- code 1.0 & 2.0 （直接看2.0的code）
https://github.com/aymericdamien/TensorFlow-Examples
- Standford公开课
CS 20SI: Tensorflow for Deep Learning Research （b站）
- TensorFlow is dead, long live TensorFlow!
https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652042006&idx=4&sn=1ffa0bf4dc8c879150aeeea1756ef938&chksm=f12187e7c6560ef1be3cea7289f66ddcf7df55647f83a9c4963d91fa323e0c3276b9824328cd&mpshare=1&scene=1&srcid=&key=3ced8d6e9f21461a80ec1844ddf4bed938b54f25fd2729db4abf80f13e433df13e9df1b00410b0caf847b0fc30577ec020cbe01da7968db26b3f107104d5c80b2db25a727414dd5c9d9d6e07f31d325e&ascene=1&uin=MjM1OTMwMzkwMA==&devicetype=Windows+7&version=62060739&lang=zh_CN&pass_ticket=k+DwQUGMalJBoJr0NlWCUv0JX+aZPg+14DPHXkP7fNoOmY6S0Zm7FygiY2Gh97fp
- 2.0 教程
https://github.com/czy36mengfei/tensorflow2_tutorials_chinese
- 官方搭建的经典模型
https://github.com/tensorflow/models
- 其它教程&code
https://github.com/machinelearningmindset/TensorFlow-Course
https://github.com/pkmital/tensorflow_tutorials
https://github.com/taki0112/Tensorflow-Cookbook
- tensorflow 2.0 快速入门 (by Chollet) 
https://threader.app/thread/1105139360226140160
- deeplearning.ai 《Introduction to TensorFlow for AI,ML and DL》
- Udacity 《Intro to TensorFlow for Deep Learning》
- 《Tensorflow 实战google深度学习框架》（应该是1.0吧，已经落后了）
- deeplearning.ai
- 莫烦python


