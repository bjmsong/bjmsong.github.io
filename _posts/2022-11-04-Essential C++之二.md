---
layout:     post
title:      Essential C++
subtitle:   之二
date:       2022-11-04
author:     bjmsong
header-img: 
catalog: true
tags:
    - C++
---
## 4. 基于对象的编程风格 Object-Based Programming
- class组成
    + 公开的操作函数和运算符
    + 私有的实现细节
- 实现一个Class
    + 所有member function都必须在class主体内进行声明，可以在类内部或者外部进行定义。如果在类内部定义，会被自动视为inline函数。
- 构造函数、析构函数
    + 构造函数
        * 函数名与类名相同
        * 不用返回任何值
        * 可以被重载，编译器会根据参数，挑选出对应的构造函数
        * 可以使用成员初始化列表的方式进行初始化
    + 析构函数
        * object结束生命时，会自动调用析构函数来释放资源
        * 析构函数并非绝对必要
        * class名称前加上"~"前缀
        * 不会有返回值
        * 没有任何参数，不可能被重载
    + 拷贝构造函数
        * 将某个class object赋值给另一个，默认会进行成员的逐一赋值操作。但是某些情况下，这种赋值会产生问题，因此需要定义拷贝构造函数，来取代默认的赋值操作。
        * 唯一参数是一个const reference，不可以重载
- mutable(可变)，const
    + member function上标注const：这个member function不会改变class object的内容
        * 编译器会进行检查
    + 成员变量标注mutable：表名对这个成员变量所做的改变不会破幻class object的常量性
- this指针
    + 指向对象的指针
    - 用途：在成员函数内用来指向其调用者
    - 实现原理：编译器自动将this指针加到每一个成员函数（静态成员函数除外）的参数表中
    - https://www.bilibili.com/video/BV1C7411Y7Kb/?spm_id_from=333.999.0.0&vd_source=7798c62f92ce545f56fd00d4daf55e26
- 静态类成员
    + static data member 用来表示唯一的、可共享的member，它可以在同一类的所有对象中被访问
    + member function只有在不访问任何non-static member的条件下才能够被声明为static
- 打造一个Iterator Class
    + 运算符（重载）函数：不用指定名称，只需要在运算符前加上operator
    + typedef
- 友元
    + 可以访问私有成员
- copy assignment operator
- function object
    + 通常function object被当作参数传递给范型算法
- 重载iostream运算符
- 指向Class Member Function的指针
    + 


## 5. 面向对象编程风格 Object-Oriented Programming
- OOP概念
    + Class的主要用途在于引入一个崭新的数据类型，能够更直接地在程序中，表现实体。
    + 继承
        * 让我们得以定义一整群互有关联的类，并共享共通的接口
        * 父类(基类)定义了所有子类(派生类)共通的公有接口和私有实现
        * 每个子类都可以增加或覆盖继承而来的东西
        * 抽象基类
            - 利用抽象基类的pointer或reference来操作系统中的各对象，而不直接操作各个实际对象。这让我们得以在不更动程序的前提下，加入或移除任何一个派生类。
    + 多态
        * 让我们得以用一种类型无关的方式来操作这些类对象
        * 基类的pointer或reference得以十分透明地指向任何一个派生类的对象
    + 动态绑定
        * 程序只有在执行时才确定应该调用哪个函数
- OOP思维
- 不带继承的多态
    + 
- 定义抽象基类
    + 找出所有子类共通的操作行为：共有接口
    + 找出哪些操作行为与类型相关：根据不同的派生类有不同的实现方式
        * 虚函数
            - 可以有定义
            - 也可以设为纯虚函数
    + 找出每个操作行为的访问层级：public，privare，protected
- 定义派生类
    + 
- 运用继承体系
    + 
- 基类应该多么抽象
    + 
- 初始化、析构、复制
    + 
- 在派生类中定义一个虚函数
    + 
- 运行时的类型鉴定机制
    + 


## 6. 以template进行编程 Programming with Templates
- 被参数化的类型
- Class Template定义
- Template类型参数的处理
- 实现一个Class Template
- 一个以Function Template完成的Output运算符
- 常量表达式与默认参数值
- 以Template参数作为一个设计策略
- Member Template Function


## 7. 异常处理 Exception Handling
- 抛出异常
    + throw
    + 异常是某种对象
- 捕获异常
    + catch
    + 匹配异常类型
    + 处理异常
    + 重新抛出异常
- 提炼异常
    + 
- 局部资源管理
    + 在异常处理机制终结某个函数之前，C++保证，函数中的所有局部对象的destructor都会被调用
    + auto_ptr  
        * 标准库提供的，使用前必须包含memory头文件
        * 是一个class template，可以实现资源的自动释放
        * 已被废弃
- 标准异常
    + 抽象基类: exception








