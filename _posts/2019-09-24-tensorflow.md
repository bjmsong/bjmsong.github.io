---
layout:     post
title:      TensorFlow
subtitle:   
date:       2019-09-24
author:     bjmsong
header-img: img/dl/tf.jpg
catalog: true
tags:
    - 深度学习
---
<ul> 
<li markdown="1"> 
![]({{site.baseurl}}/img/dl/learntf.png) 
</li> 
</ul>


### 发展历史
TensorFlow 是由 Google Brain 团队在谷歌内部第一代 DL 系统 DistBelief 的基础上改进而得到的，这一通用计算框架目前已经成为最流行的机器学习开源工具。

TensorFlow 的前身 DistBelief 是谷歌 2011 年开发的内部 DL 工具，基于 DistBelief 的 Inception 网络获得了 2014 年的 ImageNet 挑战赛冠军。虽然 DistBelief 当时在谷歌内部已经应用于非常多的产品，但它过度依赖于谷歌内部的系统架构，因此很难对外开源。经过对 DistBelief 的改进与调整，谷歌于 2015 年 11 月正式发布了开源计算框架 TensorFlow 0.5.0。相比于 DistBelief，TensorFlow 的计算框架更加通用、计算资源安排更加合理，同时支持更多的深度学习算法与平台。

2017年2月份在首届 TensorFlow 开发者大会中，谷歌正式发布了 TensorFlow 1.0。在速度上，它在 64 个 GPU 上分布式训练 Inception v3 获得了 58 倍提速。在灵活性上，TensorFlow 1.0 引入了高层 API，例如 tf.layers、tf.metrics 和 tf.losses 等模块，同时通过 tf.keras 将 Keras 库正式整合进 TF 中。

此后，TensorFlow 发布了非常多的重要更新，包括动态图机制 Eager Execution、移动端深度学习框架 TensorFlow Lite、面向 JavaScript 开发者的机器学习框架 TensorFlow.js，以及自动将 Python 转化为 TF 计算图的 AutoGraph 等。

在 TensorFlow 2.0 的规划中，Eager Execution 变为默认执行模式可能对开发者有比较大的影响，因为我们不再需要编写完整的静态计算图，并打开会话（Session）运行它。相反，与 PyTorch 一样，Eager Execution 是一个由运行定义的接口，这意味着我们在 Python 上调用它进行计算可以直接得出结果。这种方式非常符合人类直觉，因此可以预想 TensorFlow 的入门在以后会简单地多。

但是由于很多历史代码是Tensorflow1.0写的，也很有必要学习下1.0的写法，本文主要介绍Tensorflow1.0。

### 概况
TensorFlow™ 是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。
- 使用图(graph)来表示计算任务
    - 节点称之为op(operator),一个op获得0个或多个Tensor，执行计算，返回0个或多个Tensor
- 在被称之为会话(Session)的上下文(context)中执行图
    - 目的是节省资源开销
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

### 高阶API
- tf.keras
- Eager Execution
https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6

命令式编程环境，可立即评估操作，无需构建图：操作会返回具体的值，而不是构建以后再运行的计算图。
- tf.estimator
https://www.cnblogs.com/marsggbo/p/11232897.html

### 导入数据（tf.data）
#### 两个抽象类
- tf.data.Dataset : 表示一系列元素，其中每个元素包含一个或多个Tensor对象。可以通过两种不同的方式来创建数据集：
    - 创建（Dataset.from_tensor_slices,tf.data.Dataset.from_tensors），以通过一个或多个 tf.Tensor 对象构建数据集。
    - 转换（例如 Dataset.batch），以通过一个或多个 tf.data.Dataset 对象构建数据集。
- tf.data.Iterator : 消耗 Dataset 中值的最常见方法是构建迭代器对象。通过此对象，可以一次访问数据集中的一个元素（例如通过调用 Dataset.make_one_shot_iterator）。tf.data.Iterator 提供了两个操作
    - Iterator.initializer，可以通过此操作（重新）初始化迭代器的状态；
    - Iterator.get_next，此操作返回对应于有符号下一个元素的 tf.Tensor 对象。根据您的使用情形，您可以选择不同类型的迭代器。

#### 数据集结构
- 一个数据集包含多个元素，每个元素的结构都相同。一个元素包含一个或多个 tf.Tensor 对象，这些对象称为组件。每个组件都有一个 tf.DType，表示张量中元素的类型；以及一个 tf.TensorShape，表示每个元素（可能部分指定）的静态形状。

```
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
```

为元素的每个组件命名通常会带来便利性，例如，如果它们表示训练样本的不同特征。除了元组之外，还可以使用 collections.namedtuple 或将字符串映射到张量的字典来表示 Dataset 的单个元素。
```
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
```

#### 创建迭代器
构建了表示输入数据的 Dataset 后，下一步就是创建 Iterator 来访问该数据集中的元素。tf.data API 目前支持下列迭代器，复杂程度逐渐增大：
- 单次迭代器
- 可初始化迭代器
- 可重新初始化迭代器
- 可馈送迭代器

##### 单次迭代器
是最简单的迭代器形式，仅支持对数据集进行一次迭代，不需要显式初始化。单次迭代器可以处理基于队列的现有输入管道支持的几乎所有情况，但它们不支持参数化。

```
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

##### 可初始化迭代器
您需要先运行显式 iterator.initializer 操作，然后才能使用可初始化迭代器。虽然有些不便，但它允许您使用一个或多个 tf.placeholder() 张量（可在初始化迭代器时馈送）参数化数据集的定义。

```
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  assert i == value

# Initialize the same iterator over a dataset with 100 elements.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

##### 可重新初始化迭代器
可以通过多个不同的Dataset对象进行初始化。例如，您可能有一个训练输入管道，它会对输入图片进行随机扰动来改善泛化；还有一个验证输入管道，它会评估对未修改数据的预测。这些管道通常会使用不同的 Dataset 对象，这些对象具有相同的结构（即每个组件具有相同类型和兼容形状）。

```
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # Initialize an iterator over the validation dataset.
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)
```

##### 可馈送迭代器
可以与 tf.placeholder 一起使用，以选择所使用的 Iterator（在每次调用 tf.Session.run 时）（通过熟悉的 feed_dict 机制）。它提供的功能与可重新初始化迭代器的相同，但在迭代器之间切换时不需要从数据集的开头初始化迭代器。


#### 消耗迭代器中的值
Iterator.get_next() 方法返回一个或多个 tf.Tensor 对象，这些对象对应于迭代器有符号的下一个元素。每次评估这些张量时，它们都会获取底层数据集中下一个元素的值。（请注意，与 TensorFlow 中的其他有状态对象一样，调用 Iterator.get_next() 并不会立即使迭代器进入下个状态。您必须在 TensorFlow 表达式中使用此函数返回的 tf.Tensor 对象，并将该表达式的结果传递到 tf.Session.run()，以获取下一个元素并使迭代器进入下个状态。）

如果迭代器到达数据集的末尾，则执行 Iterator.get_next() 操作会产生 tf.errors.OutOfRangeError。在此之后，迭代器将处于不可用状态；如果需要继续使用，则必须对其重新初始化。
一种常见模式是将“训练循环”封装在 try-except 块中：
```
sess.run(iterator.initializer)
while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    break
```

#### 读取输入数据
- numpy

```
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```

请注意，上面的代码段会将 features 和 labels 数组作为 tf.constant() 指令嵌入在 TensorFlow 图中。这样非常适合小型数据集，但会浪费内存，因为会多次复制数组的内容，并可能会达到 tf.GraphDef 协议缓冲区的 2GB 上限。
作为替代方案，您可以根据 tf.placeholder() 张量定义 Dataset，并在对数据集初始化 Iterator 时馈送 NumPy 数组。

```
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
```

- TFRecord（tf.data.TFRecordDataset）
- 文本（tf.data.TextLineDataset）
- csv（tf.contrib.data.CsvDataset）

### 低阶API（TensorFlow Core）
#### tf.Graph : 构建计算图
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

#### tf.Session ： 运行计算图

```
sess = tf.Session()
print(sess.run(total))
```

#### TensorBoard 可视化
https://www.tensorflow.org/tensorboard/get_started
https://towardsdatascience.com/a-quickstart-guide-to-tensorboard-fb1ade69bbcf
https://medium.com/@kkoehncke/tensorboard-for-beginners-c4709998628b

#### 占位符

```
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
```

#### 数据集
**占位符适用于简单的实验，而数据集是将数据流式传输到模型的首选方法。**要从数据集中获取可运行的tf.Tensor，您必须先将其转换成 tf.data.Iterator，然后调用迭代器的 get_next 方法。

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

#### 层(tf.layers)
##### 创建层

```
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
```
##### 初始化层

```
init = tf.global_variables_initializer()
sess.run(init)
```
##### 执行层

```
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))
```

#### 特征列
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

#### 小型回归模型

```
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
_, loss_value = sess.run((train, loss))
print(loss_value)

print(sess.run(y_pred))
```

#### 张量
张量是对矢量和矩阵向潜在的更高维度的泛化。TensorFlow 在内部将张量表示为基本数据类型的 n 维数组。在编写 TensorFlow 程序时，操作和传递的主要对象是 tf.Tensor。
tf.Tensor 具有以下属性：
- 数据类型（例如 float32、int32 或 string）
- 形状

tf.Tensor 对象的阶是它本身的维数。阶的同义词包括：秩、等级或 n 维。
<ul> 
<li markdown="1"> 
![]({{site.baseurl}}/img/dl/tensor.png) 
</li> 
</ul> 

特殊张量：
- tf.Variable
- tf.constant
- tf.placeholder
- tf.SparseTensor

#### AutoGraph
AutoGraph 会在后台自动将普通的python代码转换为等效的 TensorFlow 图代码。


### 加速器
- 分布策略
- GPU
- TPU


### 参考资料
- 教程（tf1.x，tf2.x）
  - https://github.com/aymericdamien/TensorFlow-Examples
  - https://www.zhihu.com/question/49909565
- https://tensorflow.google.cn/guide/
- deeplearnig.ai
    - https://mooc.study.163.com/university/deeplearning_ai#/c
    - deeplearning.ai 1.0&2.0课程
    - 课件&课后作业：
    https://blog.csdn.net/u013733326/article/details/83341643
    https://github.com/Wasim37/deeplearning-assignment
    - 笔记 
        - https://github.com/fengdu78/deeplearning_ai_books
        - https://zhuanlan.zhihu.com/p/35333489?utm_source=wechat_session&utm_medium=social&utm_oi=72535160913920&from=singlemessage
        - https://github.com/stormstone/deeplearning.ai
- https://github.com/ageron/handson-ml
    - +pdf
- 《Deep Learning with Python》
https://livebook.manning.com/book/deep-learning-with-python/
https://github.com/fchollet/deep-learning-with-python-notebooks
- https://mp.weixin.qq.com/s?__biz=Mzg5ODAzMTkyMg==&mid=2247487433&idx=1&sn=403cac1730a04c967e99fbb44c4aeae1&chksm=c0698794f71e0e82dabf12268746f850686565ac8caa56c24677aa2def746fa6456bbc49e8c1&mpshare=1&scene=1&srcid=&sharer_sharetime=1573605136825&sharer_shareid=602b1ccf63ca4ea52755ecd058f6d407&key=7d1e4c0a06963a8b5ff47884a1e90d581015f29f3d3c76b231629d05c0e9a8ee7afc3ecd72e9f8a0360af9218e2e8b2ef591aa85509f0bc511c70cfd802f27ca1c2f7aa13cb606259b5643fba475de8d&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+7&version=62070152&lang=zh_CN&pass_ticket=gd9xEtuY4W21P%2BXGRqDGswdbHvyZZJTrRy6smCdZoYMTxRtU7jUmWmcuzbnc2Z%2Bb





