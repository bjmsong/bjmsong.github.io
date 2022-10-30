---
layout:     post
title:      Essential C++
subtitle:   
date:       2022-08-21
author:     bjmsong
header-img: 
catalog: true
tags:
    - C++
---
## code
https://github.com/bjmsong/essentialCpp

## C++编程基础 Basic C++ Programming
- class：用户自定义的数据类型（user-defined data type），增强类型抽象化的层次
    - class定义应该分为两部分
        - 头文件（header file）：用来声明该class的各种操作行为
        - 代码文件（program text）：包含了操作行为的具体实现
- 对象的定义和初始化
    - 对象初始化方式
        - 使用=运算符，如int num_tries = 0
        源自C语言
        - 构造函数语法，如int num_right(0)
        解决“多值初始化”问题
        使内置类型与class类型的初始化方式得到统一
- 条件、循环
- string vs C-style 字符串(char s[])
    - 建议用string
    - string对象会动态地随字符串长度的增加而增加其存储空间，C-style字符串只能分配固定的空间
    - C-style字符串不记录自身长度，strlen()
- Array，Vector
    - 建议用vector
    - array的大小必须固定，vector可以动态地随着元素的插入而扩展存储空间
    - array并不存储自身大小，必须考虑对它的访问可能导致溢出
- 指针
- 文件的读写
    - 头文件：fstream
    - 输出文件：ofstream
    - 读取文件：ifstream
    - 读写文件：fstream

## 面向过程的编程风格 Procedural Programming
- 编写函数
    - 先声明，再调用
    - 参数校验
    - 返回值用bool
        - False: 发生异常，没有返回用户想要的结果
        - True：正确返回
    - return作用
        - 用在有返回值的函数中，向主调函数返回一个值
        - 用在无返回值的函数中，提交结束函数的执行
- 调用函数
    - 传址(by reference): 参数声明为一个reference，对reference的所有操作都和面对“reference所代表的对象”所进行的操作一般无二。
        - 优点
            - 可以直接对所传入的对象进行修改
            - 降低复制大型对象的额外负担
        - 也可以用指针，但是用法不同，而且用指针的话要做空指针校验
    - 传值(by value): 传入的对象会被复制一份，原对象和副本之间没有任何关系，函数内部的改动不会影响原对象
    - 调用函数时，会在内存中建立起一块特殊区域，成为"程序堆栈"。这块特殊区域提供了每个函数参数的存储空间，也提供了函数所定义的每个对象的内存空间。一旦函数完成，这块内存就会被释放掉，或者说是从程序堆栈中被pop出来。
    - 作用域：对象在程序内的存活区域
        - 系统自动管理
            - local scope ： 函数内声明
            - file scope ： 函数外声明
        - 程序员自行管理
            - dynamic extent，heap memory
            - new, delete
            - 内存泄露
- 默认参数
    - 以参数传递作为函数间的沟通方式，比直接将对象定义于file scope更适当
        - 函数如果过度依赖定义于file scope内的对象，就比较难以在其他环境中重用，也比较难以修改
    - 默认值只能指定一次，一般在函数声明处而非定义处
- 局部静态对象
    - 局部静态对象所处的内存空间，即使在不同的函数调用过程中，依然持续存在
- inline
    - 将函数声明为inline，表示要求编译器在每个函数调用点上，将函数的内容展开，使我们获得性能改善。
    - 适合声明为inline的函数：体积小、经常被调用，计算逻辑不复杂
    - inline函数的定义，常常被放在头文件中
- 重载函数
    - 参数列表不相同，函数名相同
    - 无法根据返回值类型来区分两个具有相同名称的函数
- 模板函数
    - 将参数列表的部分（或全部）参数的类型信息抽离出来
    - 具体的类型信息在采用function template具体实例时指定
- 函数指针
    - 指明其所指函数的返回类型及参数列表
- 设定头文件
    - 定义只有一份，声明可以有多份


## 泛型编程风格 Generic Programming
- STL: Standard Template Library
    - 容器(container): vector,list,set,map....
    - 泛型算法: find(),sort(),replace(),merge()....
        - 与操作对象的类型相互独立：通过function template技术
        - 与操作的容器相互独立：不直接在容器身上进行操作，而是借由一对iterator(first和last),标示要进行迭代的元素范围
- 指针的算术运算
    - 目标：设计一个函数可以同时处理vector/array/list内的任意类型元素
        - e.g.: find()
    - 切分成子问题
        - 将array的元素传入find(),而非指明该array
            - 当数组被传给函数，或是由函数中返回，仅有第一个元素的地址会被传递
            - 直接传指针更清晰，并且仍然可以通过下标运算符访问array的每个元素
        - 将vector的元素传入find(),而非指明该vector
            - 跟array不同的是，vector要先校验是否为空
        - list：链表
- Iterator(泛型指针)
- 所有容器的共通操作
- 顺序性容器
- 泛型算法
- Map
- Set
- Iterator Inserter
- iostream Iterator



## 基于对象的编程风格 Object-Based Programming
- 
- 


## 面向对象编程风格 Object-Oriented Programming
-


## 以template进行编程 Programming with Templates
-


## 异常处理 Exception Handling
- 


## 参考资料
