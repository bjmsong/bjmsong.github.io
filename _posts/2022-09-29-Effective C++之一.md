---
layout:     post
title:      Effective C++
subtitle:   改善程序与设计的55个具体做法之一
date:       2022-09-29
author:     bjmsong
header-img: 
catalog: true
tags:
    - C++
---
## 导读
- 声明(declartion): 告诉编译器某个东西的名称和类型，但略去细节
    + 函数签名(signature): 参数和返回类型
- 定义(definition): 提供编译器一些声明式所遗漏的细节，为对象开辟内存
- 初始化(initialization): 给予对象初值
    + 对于用户自定义对象，初始化由构造函数执行
        * default构造函数：可被调用而不带任何参数
        * 将构造函数声明为explicit：阻止它们被用来执行隐式类型转换
        * copy构造函数：以同型对象初始化自我对象
            - 定义一个对象如何passed by value
            - copy assignment操作符：从另一个同型对象中拷贝其值到自我对象
- STL(标准模板库)
    + 容器、迭代器、算法
    + 许多相关能力以funcion object实现：行为像函数的对象
- 避开不明确行为(undefined behavior): 对一个null指针取值，索引超过数组大小等
- interface(接口): C++的"接口"一般指的是函数的签名或class的可访问元素
## 让自己习惯C++
### 1. 视C++为一个语言联邦
- 多重范型编程语言 multiparadigm programming language
    + 过程形式(procedural)
    + 面向对象形式(object-oriented)
    + 函数形式(functional)
    + 泛型形式(generic)
    + 元编程形式(metaprogramming)
- 由相关语言组成的联邦
### 2. Prefer const,enum and inline to #define
- #define存在的问题
- 

### 3. 尽可能使用const
- 
- 

### 4. Make sure that objects are initialized before they're used

## 构造/析构/赋值运算 Constructors,Destructors, and Assignment Operators
### 5. 了解C++

### 6.

### 7.

### 8.


### 9.

### 10.

### 11.

### 12.

### 13.

### 14.

### 15.

### 16.

### 17.

### 18.


## 
### 48. 模板(template)元编程


## 参考资料
- https://blog.csdn.net/fengbingchun/article/details/102761542

