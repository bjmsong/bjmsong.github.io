---
layout:     post
title:      Welcome to C++
subtitle:   
date:       2022-04-04
author:     bjmsong
header-img: img/
catalog: true
tags:
    - 
---
## 
- visual studio
    - 模式：debug, release
- 编译(compiling)
    - 只编译：visual studio(编译/ctrl+F7)
    - .cpp>.obj(二进制文件,机器码)
    - 预处理：#include, #defind, #ifdef
    - 标记解释，解析 => 生成抽象语法树
    - 编译器优化
- 链接(linking）
    - 编译+链接：visual studio(build/F5)
    - .obj->.exe（可执行二进制文件）
    - 把编译过程中生成的所有对象文件链接起来
        -  找到每个符号和函数的位置，并将它们链接在一起
    - 如果只有一个cpp文件也需要链接，因为需要知道入口点（可设置，  可以不是main函数）在哪 
    - static: 链接只发生在该文件的内部 
    - inline：用函数的body取代调用
    - 静态链接、动态链接
- 变量
    - 存储在内存中：堆 or 栈
    - 原始数据类型
        - 不同类型之间唯一的区别就是大小（占用多少内存）
        - char,short,int,long,float,double,bool
        - unsigned
    - 数据类型的实际大小取决于编译器
        - int：4个字节，存储范围是 -2^31~2^31
    - sizeof: 查看变量占用内存大小
- 函数
    - 
- 头文件
    - 声明(declaration)
    - 可以有函数body
- 定义（definition）
    - 声明+函数body
- 指针
- 引用
- 

## 参考资料
