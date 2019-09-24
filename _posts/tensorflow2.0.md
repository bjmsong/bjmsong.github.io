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



- code 1.0 & 2.0 
https://github.com/aymericdamien/TensorFlow-Examples
- Standford公开课
CS 20SI: Tensorflow for Deep Learning Research （b站）
- 2.0 教程
https://github.com/czy36mengfei/tensorflow2_tutorials_chinese
- 官方模型
https://github.com/tensorflow/models
- 其它
https://github.com/machinelearningmindset/TensorFlow-Course
https://github.com/pkmital/tensorflow_tutorials
https://github.com/taki0112/Tensorflow-Cookbook


- tensorflow 2.0 快速入门 (by Chollet) 
https://threader.app/thread/1105139360226140160
- deeplearning.ai 《Introduction to TensorFlow for AI,ML and DL》
- Udacity 《Intro to TensorFlow for Deep Learning》
- 《Tensorflow 实战google深度学习框架》


