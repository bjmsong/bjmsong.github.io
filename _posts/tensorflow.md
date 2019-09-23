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
## 高阶API（官方建议使用）
- tf.keras
- tf.estimator
- Eager Execution


## 导入数据
tf.data


## 低阶API（TensorFlow Core）
- tf.Graph : 构建计算图
    - 操作(op)
    - 张量(tf.Tensors): 不具有值，它们只是计算图中元素的手柄
```
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
```
- tf.Session ： 运行计算图
```
sess = tf.Session()
print(sess.run(total))
```

- TensorBoard ：将计算图可视化
- 占位符
```
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
```
- 数据集
占位符适用于简单的实验，而数据集是将数据流式传输到模型的首选方法。要从数据集中获取可运行的tf.Tensor，您必须先将其转换成 tf.data.Iterator，然后调用迭代器的 get_next 方法。
```
my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()
```

- 层(tf.layers)
1. 创建层
```
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
```
2. 初始化层
```
init = tf.global_variables_initializer()
sess.run(init)
```
3. 执行层
```
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))
```

- 特征列
tf.feature_column.input_layer，此函数只接受密集列作为输入，因此要查看类别列的结果，您必须将其封装在 tf.feature_column.indicator_column 中。
```
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)
```

## 加速器
- 分布策略
- GPU
- TPU


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

## tensorflow-serving

## 参考资料
- 官方教程
https://tensorflow.google.cn/tutorials/
- deeplearning.ai
- github热门教程
https://github.com/aymericdamien/TensorFlow-Examples




CS 20SI: Tensorflow for Deep Learning Research （b站）
- https://github.com/machinelearningmindset/TensorFlow-Course
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
tenserboard
