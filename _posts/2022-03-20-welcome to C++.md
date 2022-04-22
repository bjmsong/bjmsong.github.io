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
- 编译(compiling)
    - 只编译：visual studio(编译/ctrl+F7)
    - .cpp>.obj(二进制文件,机器码)
    - 预处理
        - #include： 就是copy&paste
            - 方括号只用于编译器的include路径，引号用于所有
            - C标准库里的头文件一般都有.h扩展名，而C++标准库的头文件没有
        - #define, #ifdef
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
    - 主要目的：防止代码重复，好维护
    - 调用函数有额外的开销，除非是内联(inline)函数 
    - 声明(declaration)：在头文件中
    - 定义（definition，声明+函数body）：在cpp文件/翻译单元中
- 头文件
    - 存放声明，而非定义
        - 只能定义函数一次
    - 可以通过include头文件的方式，避免在cpp文件中写大量声明
    - #pragma once：防止把单个头文件多次include到一个cpp/翻译单元里
        - 头文件保护符
        - 之前解决这个问题的方式：#ifndef #define #endif
- debug in visual studio
    - de-bug：清除bug
    - debug模式
        - 会做额外的事情来提高debug效率，因此会减慢程序运行的速度
        - release模式用来发布
    - 电脑几乎总是对的 
    - breakpoints(断点)，reading memory：暂停程序，看看内存中发生了什么
    - continue：运行直到下一个断点
    - step into：进入当前函数
    - step over：调到下一行 
    - step out: 跳出当前函数
    - 鼠标悬停到变量上，可以显示变量当前值
    - 内存视图：内存地址、实际值（16进制表示）、对应的ASCII解释
        - 2个16进制=1字节 
- conditionals/if statements and branches
    - slow the program down
- loops
    - for, while
    - do-while 
- control flow
    - continue: go to the next iteration of the loop
    - break: get out of the loop
    - return: get out of the function 
- 指针(raw Pointers)
    - is a integer number which stores a memory address 
    - memory in computer: like one-dimension line
    - type of pointer：the type of the data in that address
    - nullptr
    - 取址 &
    - 
- smart Pointers
    - 
    - 
- 引用(References)
    - &a：变量a的内存地址
    - 
- Class
    - 
- Class vs Struct
- static
- 回调函数

## 参考资料
