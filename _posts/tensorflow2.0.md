---
layout:     post
title:      TensorFLow 2.0
subtitle:   
date:       2019-10-08
author:     bjmsong
header-img: img/dl/tf2.0.png
catalog: true
tags:
    - 深度学习
---
> from 官方教程

### tf.keras
tf.keras已经成为tensorflow的一个高级API，keras有sequential和functional两种API方式。

#### sequential API
可以像搭积木一样搭建神经网络，搭建流程是这样的
```
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
```

#### functional and subclassing APIs 
可以更灵活的构建定制化模型，比如：多个输入和输出，共享网络层等
```
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

# prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# build model
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# train & test
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))

```


### 数据加载和预处理
可以将各种格式数据加载进tf.data.Dataset，包括：csv，numpy，pandas.DataFrame等。

### Estimators
Estimators是TensorFlow的高阶API，
特征列是原始数据和 Estimator 之间的媒介， 特征列内容丰富，使您可以将各种原始数据转换为 Estimator 可以使用的格式，从而可以轻松地进行实验。




### 官方模型和数据集
>[官方模型和数据集](https://tensorflow.google.cn/resources/models-datasets)

#### 官方模型样例
[官方模型](https://github.com/tensorflow/models/tree/master/official)使用的是TensorFLow的高阶API。这些模型可读性好，紧跟最新的Tensorflow API，被良好维护，经过官方测试，同时性能上也做了很好的优化。提供的模型有：
- nlp：bert、transformer， xlnet
- cv：resnet
- recommendation：ncf
- 还有一些模型不更新到TensorFlow 2.x（在R1文件夹中）：boosted_trees, wide&deep

另外还有一大波模型正在袭来：GPT2，更多的transforme，、EfficientNet, MnasNet。。。

#### Tensorflow Hub
[TensorFlow Hub](https://tensorflow.google.cn/hub/)是一个通过复用机器学习模块可用于迁移学习的库，致力于TensorFlow机器学习模型的组件化。迁移学习可以：
- 使用较小的数据集训练模型，
- 改善泛化效果，以及
- 加快训练速度。
TensorFlow Hub是一个共享可重用机器学习模型的平台，其愿景是为研究人员和开发人员提供一种方便的方式分享他们的工作。可以通过Tensorflow Hub直接导入已经训练好的模型。


#### 数据集
- [TensorFlow官方数据集](https://tensorflow.google.cn/datasets/)：可用于TensorFlow的一系列数据集的集合, 所有数据集都显示为tf.data.Datasets，可以提供易于使用且具有高性能的输入流水线。
- Google研究数据集
- 其他数据集资源


### 工具
- Colab：免费的Jupyter笔记本环境，不需要任何设置就可以使用，可以在浏览器中方便地执行Tensorflow代码。
- [Tensorboard](https://tensorflow.google.cn/tensorboard)：一套可视化工具，用于理解、调试和优化TensorFlow程序。
- [What-if](https://pair-code.github.io/what-if-tool/):一种以无代码方式探究机器学习模型的工具，对模型的理解、调试和公平性很有帮助，可以在TensorBoard或Colab中使用。
- 
- 
- TensorFlow PlayGround

### TensorFlow 2.0变化
- 删除了冗余API，部分API被2.0的新API替代，还有参数变化等。**可以使用官方提供的[升级脚本](https://tensorflow.google.cn/guide/upgrade)自动升级1.0代码**。
- Eager Execution 变为 TensorFlow 2.0 默认的执行模式。这意味着 TensorFlow 如同 PyTorch 那样，由编写静态计算图全面转向了动态计算图。不再需要编写完整的静态计算图，并打开会话（Session）运行它。
    - Eager 模式变为默认设置之后，开发者可以在原型搭建完成后，利用 AutoGraph 把在 Eager 模式下搭建的模型自动变成计算图。
- No more globals（？）
- session被function替代，AutoGraph可以把Eager模式下搭建的模型自动变成计算图
```
# TensorFlow 1.X
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# TensorFlow 2.0
outputs = f(input)
```
    

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
https://blog.csdn.net/shin627077/article/details/78592729
https://blog.csdn.net/heyc861221/article/details/80129169
https://www.jianshu.com/p/2fffd0e332bc

### 参考资料
- https://tensorflow.google.cn/tutorials
- https://blog.csdn.net/mogoweb/article/details/97722478



- https://tensorflow.google.cn/guide
- https://github.com/snowkylin/tensorflow-handbook
- tensorflow 2.0 快速入门 (by Chollet) 
https://threader.app/thread/1105139360226140160
- https://github.com/aymericdamien/TensorFlow-Examples
- https://medium.com/tensorflow/effective-tensorflow-2-0-best-practices-and-whats-changed-a0ca48767aff
- TensorFlow is dead, long live TensorFlow!
https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652042006&idx=4&sn=1ffa0bf4dc8c879150aeeea1756ef938&chksm=f12187e7c6560ef1be3cea7289f66ddcf7df55647f83a9c4963d91fa323e0c3276b9824328cd&mpshare=1&scene=1&srcid=&key=3ced8d6e9f21461a80ec1844ddf4bed938b54f25fd2729db4abf80f13e433df13e9df1b00410b0caf847b0fc30577ec020cbe01da7968db26b3f107104d5c80b2db25a727414dd5c9d9d6e07f31d325e&ascene=1&uin=MjM1OTMwMzkwMA==&devicetype=Windows+7&version=62060739&lang=zh_CN&pass_ticket=k+DwQUGMalJBoJr0NlWCUv0JX+aZPg+14DPHXkP7fNoOmY6S0Zm7FygiY2Gh97fp
- https://github.com/czy36mengfei/tensorflow2_tutorials_chinese
- 教程&code
https://github.com/machinelearningmindset/TensorFlow-Course
https://github.com/pkmital/tensorflow_tutorials
https://github.com/taki0112/Tensorflow-Cookbook
- deeplearning.ai 《Introduction to TensorFlow for AI,ML and DL》
- Udacity 《Intro to TensorFlow for Deep Learning》
- 《Tensorflow 实战google深度学习框架》（应该是1.0吧，已经落后了）
- 
https://mp.weixin.qq.com/s?__biz=MzIzNjc1NzUzMw==&mid=2247530867&idx=1&sn=e0855bd03cbedde94df8cf947b9d8a68&chksm=e8d0ce01dfa74717ab43f773af0c5add8c8b7869ab13549f7f3eae6200faf1c2d739b7ac398c&mpshare=1&scene=1&srcid=&sharer_sharetime=1569996302466&sharer_shareid=49581f7bdbef8664715f595bc62d7044&key=8a4bbb55c6c79ce6fa454f5f47b52efb672dd9d3c51b557d7e3f94b262546f8efda62a6b5db95a39bdf13e98095ff283349bebd5031578f6e674caa30dc82acb1bc085b293576cb51894c84e64f645f0&ascene=1&uin=MjM1OTMwMzkwMA%3D%3D&devicetype=Windows+10&version=62070141&lang=en&pass_ticket=doB3BRar8EovetMHBhmSUx44fADeRFqbXn4STwtBWkUO4U8Pw6IebWbKuRSVBN6v
- https://mp.weixin.qq.com/s/EUICEJ-LTY28N6RTrp5HSg

