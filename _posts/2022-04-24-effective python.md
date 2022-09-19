---
layout:     post
title:      effective python
subtitle:   
date:       2022-04-24
author:     bjmsong
header-img: 
catalog: true
tags:
    - 
---
## 用Pythonic方式来思考
2. 遵循PEP8风格指南
3. 了解bytes,str与unicode的区别
    - 在python3中，bytes是一种包含8位值的序列，str是一种包含Unicode字符的序列
    - 把编码和解码操作放在界面最外围来做，程序的核心部分应该使用Unicode字符类型(python3中的str)
4. 用辅助函数来取代复杂的表达式
5. 了解序列切割的方法
    - 在切割后得到的新列表上进行修改，不会影响原列表
6. 在单次切片操作内，不要同时指定start,end和stride
7. 用列表推导来取代map和filter
    - 字典(dict)和集(set)，也有和列表类似的推导机制
8. 不要使用含有两个以上表达式的列表推导
9. 用生成器表达式来改写数据量较大的列表推导
    - 由生成器表达式所返回的迭代器，可以逐次产生输出值，从而避免了内存用量问题
    - 把某个生成器表达式所返回的迭代器，放在另一个生成器表达式的for子表达式中，即可将二者组合起来
    - 串在一起的生成器表达式执行速度很快
10. 尽量使用enumerate取代range
    - enumerate函数提供了一种精简的写法，可以在遍历迭代器时获得每个元素的索引
11. 用zip函数同时遍历多个迭代器
13. 合理利用try/except/else/finally结构中的每个代码块
    - 无论try块是否发生异常，都可利用try/finally复合语句中的finally块来执行清理工作
    - else块可以用来缩减try块中的代码量，并把没有发生异常时所要执行的语句与try/except代码块隔开
    - 顺利运行try块后，若想使某些操作能在finally块的清理代码之前执行，则可将这些操作写到else块中

## 函数
14. 尽量用异常来表示特殊情况，使得调用者可以正确处理
16. 考虑用生成器来改写直接返回列表的函数
18. 用数量可变的位置参数减少视觉干扰
19. 用关键字参数来表达可选的行为

## 类与继承
22. 尽量使用辅助类来维护程序的状态，而不要使用字典和元组
    - 不要使用包含其他字典的字典，也不要使用过长的元组
    - 用来保存程序状态的数据结构一旦变得过于复杂，就应该将其拆解为类，以便提供更为明确的接口，并更好地封装数据
        - 代码量会增加，但是更容易理解，更易扩展
    - 如果容器中包含简单而又不可变的数据，那么可以先用namedtuple来表示，待稍后有需要时，再修改为完整的类
23. 简单的接口应该接受函数，而不是类的实例
    - 
24. 


## 元类及属性


## 并发及并行


## 内置模块 


## 