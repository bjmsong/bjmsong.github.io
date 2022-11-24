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
- 避开不明确行为(undefined behavior): 例如对一个null指针取值，索引超过数组大小等
- interface(接口): C++的接口一般指的是函数的签名或class的可访问元素

## 一. 让自己习惯C++
### 1. 视C++为一个语言联邦
- 多重范型编程语言 multiparadigm programming language
    + 过程形式(procedural)
    + 面向对象形式(object-oriented)
    + 函数形式(functional)
    + 泛型形式(generic)
    + 元编程形式(metaprogramming)
- 由相关语言组成的联邦
    + C
    + Object-Oriented C++
    + Template C++：泛型编程
    + STL
- C++高效编程守则视状况而变化，取决于你使用C++哪一部分

### 2. Prefer const,enum and inline to #define
- #define存在的问题
    + 定义的名称也许没有进入记号表(symbol table)，会给调试带来困惑
    + 可能导致目标码出现多份
- 对于单纯常量，最好以const对象或enums替换#define
- 对于形似函数的宏(macros)，最好改用inline函数替换#define

### 3. 尽可能使用const
- 将某些东西声明为const可帮助编译器侦测出错误用法。const可被施加于任何作用域内的对象、函数参数、函数返回值、成员函数
    + 面对指针，可以指针自身、指针所指对象、两者都是（或都不是）const
    + const_iterator: 迭代器所指向的东西不可被改动
    + 函数返回值是const：可以降低调用者因意外造成的错误
    + 函数参数：一般建议用const，除非需要改动参数
    + const成员函数：const对象可以调用
- 编译器强制实施bitwise constness，但你编写程序时应该使用conceptual constness
    + bitwise constness：成员函数只有在不更改对象的任何成员变量(static除外)时才可以说是const
    + conceptual constness：
- 当const和non-const成员函数有着实质等价的实现时，令non-const版本调用const版本可避免代码重复

### 4. Make sure that objects are initialized before they're used
- 

## 二. 构造/析构/赋值运算 Constructors,Destructors, and Assignment Operators
### 5. 了解C++默默编写并调用哪些函数

### 6. 若不想用编译器自动生成的函数，就应该明确拒绝

### 7. 为多态基类声明virtual析构函数

### 8. 别让异常逃离析构函数


### 9. 绝不在构造和析构过程中调用virtual函数

### 10. 令operator=返回一个reference to \*this

### 11. 在operator=中处理“自我赋值”

### 12. 复制对象时勿忘其每一个成分

## 三. 资源管理
### 13. 用对象管理资源

### 14. 在资源管理类中小心coping行为

### 15.

### 16.

### 17.

## 四. 设计与声明
### 18. 让接口容易被正确使用，不易被误用

### 19. 设计class犹如设计type

### 20. Prefer pass-by-inference-to-const to pass-by-value

### 21. Don't try to return a reference when you must return an object

### 22. 将成员变量声明为private

### 23. Prefer non-member、non-friend functions to member functions

### 24. Declare non-member functions when type conversions should apply to all parameters

### 25. 考虑写出一个不抛异常的swap函数

## 五. 实现
### 26. 尽可能延后变量定义的时间

### 27. Minimize casting

### 28.

### 29.

### 30.

### 31. 将文件间的编译依存关系降到最低

## 六. 继承与面向对象设计
### 32. Make sure your public inheritance models "is-a"

### 33. 避免遮掩继承而来的名称

### 34. 区分接口继承和实现继承

### 35.

### 36.

### 37.

### 38.

### 39.

### 40. 明智而审慎地使用多继承

## 七. 模板与泛型编程
### 41.

### 42.

### 43.

### 44.

### 45.

### 46.

### 47.

### 48. 模板(template)元编程

## 八. 定制new和delete
### 49.

### 50.

### 51.

### 52.

## 九. 其它
### 53.

### 54.

### 55.



## 参考资料
- 中文pdf
    + http://aleda.cn/books/Effective_C++.pdf
- 英文pdf
    + https://github.com/GunterMueller/Books-3/blob/master/Effective%20C%2B%2B%203rd%20ed.pdf
    + https://www.dsi.fceia.unr.edu.ar/downloads/informatica/info_II/c++/Effective%20C++%20+%20More%20Effective%20C++.pdf
- https://blog.csdn.net/fengbingchun/article/details/102761542

