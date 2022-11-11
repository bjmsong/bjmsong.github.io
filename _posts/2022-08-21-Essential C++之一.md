---
layout:     post
title:      Essential C++
subtitle:   之一
date:       2022-08-21
author:     bjmsong
header-img: 
catalog: true
tags:
    - C++
---
## 官方提供的源代码
https://www.informit.com/store/essential-c-plus-plus-9780201485189

## 1. C++编程基础 Basic C++ Programming
- 如何撰写C++程序
    + class：用户自定义的数据类型（user-defined data type）
        * 增加程序内类型抽象化的层次
        * class定义分为两部分：
            - 头文件（header file）：用来声明该class的各种操作行为
            - 代码文件（program text）：包含了操作行为的具体实现
    + 字符常量(character literal): 由单引号括住，分为可打印字符和不可打印字符(如换行符 '\n')
- 对象的定义和初始化
    - 对象初始化方式
        - 使用赋值运算符(=)，如int num_tries = 0
            + 源自C语言
        - 构造函数语法，如int num_right(0)
            + 解决“多值初始化”问题
            + 使内置类型与class类型的初始化方式得到统一
    - 被定义为const的对象，在获得初值之后，无法再有任何变动
- 条件、循环
- string vs C-style 字符串(char s[])
    - 建议用string
    - string对象会动态地随字符串长度的增加而增加其存储空间，C-style字符串只能分配固定的空间
    - C-style字符串不记录自身长度，strlen()
- Array，Vector
    + array
        * 需要指定大小，array的大小必须是个常量表达式
        
        ```cpp
        const int seq_size = 18;
        int pell_seq[seq_size];

        // 初始化列表，vector不支持这种方式
        pell_seq = {1,2,3}
        ```

    + vector
        
        ```cpp
        #include <vector>
        vector<int> pell_seq (seq_size);  // 泛型，seq_size不一定得是个常量表达式

        // 可以利用一个已初始化的array作为vector的初值

        // vector可以知道自己大小是多少
        pell_seq.size()
        ```

    + 建议用vector
        * array的大小必须固定，vector可以动态地随着元素的插入而扩展存储空间
        * array并不存储自身大小，必须考虑对它的访问可能导致溢出
- 指针
    + 在解引用操作之前，要校验指针是否为空指针
- 文件的读写
    - 头文件：fstream
    - 输出文件：ofstream
    - 读取文件：ifstream
    - 读写文件：fstream

## 2. 面向过程的编程风格 Procedural Programming
- 编写函数
    - 先声明，再调用
        + 声明必须提供返回类型、函数名、参数列表
        + 声明让编译器得以检查后续出现的使用方法是否正确
    - 对参数进行校验
    - 返回值来标识函数返回是否正确
        - False: 发生异常，没有返回用户想要的结果
        - True：正确返回
    - 实际要修改的变量可以通过通过引用的方式作为形参传入
    - return作用
        - 用在有返回值的函数中，向主调函数返回一个值
        - 用在无返回值的函数中，提交结束函数的执行
- 调用函数
    - 传址(by reference): 参数声明为一个reference，对reference的所有操作都和面对“reference所代表的对象”所进行的操作一般无二。
        - 优点
            - 可以直接对所传入的对象进行修改
            - 降低复制大型对象的额外负担
        - 也可以用指针，但是用法不同(指针比较复杂)，而且用指针的话要做空指针校验
    - 传值(by value): 传入的对象会被复制一份，原对象和副本之间没有任何关系，函数内部的改动不会影响原对象
    - 调用函数时，会在内存中建立起一块特殊区域，成为"程序堆栈"。这块特殊区域提供了每个函数参数的存储空间，也提供了函数所定义的每个对象的内存空间。一旦函数完成，这块内存就会被释放掉，或者说是从程序堆栈中被pop出来。
    - 作用域：对象在程序内的存活区域
        + 函数内定义的对象（除了static对象），只存在于函数执行期间。如果将这些局部对象的地址返回，会导致运行时错误。
        + scope: 对象在程序内的存活区域
        + 系统自动管理
            * local scope ：函数内声明
            * file scope ：函数外声明
        + 程序员自行管理：动态内存管理
            - dynamic extent，heap memory
            - new, delete
            - 内存泄露：heap分配而来的对象没有被释放
- 默认参数
    - 以参数传递作为函数间的沟通方式，比直接将对象定义于file scope更适当
        - 函数如果过度依赖定义于file scope内的对象，就比较难以在其他环境中重用，也比较难以修改
    - 默认值只能指定一次，一般在函数声明处而非定义处
- 局部静态对象(local static object)
    - 局部静态对象所处的内存空间，即使在不同的函数调用过程中，依然持续存在
- inline函数
    - 将函数声明为inline，表示要求编译器在每个函数调用点上，将函数的内容展开，使我们获得性能改善。
    - 适合声明为inline的函数：体积小、经常被调用，计算逻辑不复杂
    - inline函数的定义，常常被放在头文件中
- 重载函数
    - 参数列表不相同，函数名相同
    - 编译器无法根据返回值类型来区分两个具有相同名称的函数
- 模板函数
    - 将参数列表的部分（或全部）参数的类型信息抽离出来
    - 具体的类型信息在采用function template具体实例时指定
    - 模板函数同时也可以是重载函数
- 函数指针
    - 必须指明其所指函数的返回类型及参数列表
- 头文件
    + 头文件可以避免重复声明
    + 定义只有一份，声明可以有多份
        * inline是例外，inline的定义必须放在头文件中
    + 双引号引入
        * 头文件和包含此文件的程序代码文件位于同一磁盘目录下
        * 被认为是用户提供的头文件
    + 尖括号来引入
        * 头文件和包含此文件的程序代码文件在不同磁盘目录下
        * 此文件被认定为标准的或项目专属的头文件，编译器会在默认的磁盘目录中寻找


## 3. 泛型编程风格 Generic Programming
- STL(Standard Template Library)主要由两种组件构成：
    - 容器(container)
        + 顺序性容器：vector, list
            * 主要进行迭代操作
        + 关联容器：set, map
            * 可以快速查找元素
    - 泛型算法: find(),sort(),replace(),merge()....
        - 与操作对象的类型相互独立：通过function template(函数模板)技术
        - 与操作的容器相互独立：不直接在容器身上进行操作，而是借由一对iterator(first和last)，标示要进行迭代的元素范围
- 指针的算术运算
    - 目标：设计一个函数可以同时处理任意容器(vector/array/list)的元素
    - 切分成子问题
        - 将array的元素传入find(),而非指明该array
            - 当数组被传给函数，或是由函数中返回，仅有第一个元素的地址会被传递
            - 直接传指针更清晰，并且仍然可以通过下标运算符访问array的每个元素
        - 将vector的元素传入find(),而非指明该vector
            - 跟array不同的是，vector要先校验是否为空
        - list：链表
            + 指针的算术运算不适用与链表，因为内存空间不连续
- Iterator(泛型指针)
    + 提供一层抽象，把底层指针的处理都放在这层抽象中，让用户无须直接面对指针进行操作，同时可以提供“和指针相同的语法”
    + 每个标准容器都提供begin()/end()函数，可以返回一个iterator
    
    ```cpp
    vector<string> svec;
    vector<string>::iterator iter = svec.begin();
    ```
    
- 所有容器(所有容器类以及string类)的共通操作
    + ==,!=
    + 赋值(=)
    + empty()
    + size()
    + clear()
    + begin(),end()
    + insert()
    + erase()
- 顺序性容器
    + 维护一组排列有序、类型相同的元素
    + vector
        * 以一块连续内存来存放元素
        * 查找效率高（因为每个元素都被存储在距离起始点的固定偏移位置上），插入/删除效率低
    + list
        * 双向链表
        * 随机访问效率不高，插入/删除效率高
    + deque
        * 以连续内存来存放元素
        * 对于两端元素的插入和删除，效率较高
    
    ```cpp
    #include <vector>
    #include <list>
    #include <deque>

    // 定义顺序性容器
    // 方法一：空容器
    list<string> slist;
    vector<int> ivec;

    // 方法二：产生特定大小的容器，每个元素初始化为默认值
    list<int> slist(1024);
    vector<string> svec(32);

    // 方法三：产生特定大小的容器，并指定初始值
    vector<int> ivec(10, -1);
    list<string> slist(16, "unassigned");

    // 方法四：通过一对iterator产生容器
    int ia[8] = {1,1,2,3,5,8,13,21};
    vector<int> fib(ia, ia+8);

    // 方法五：通过容器产生新容器
    list<string> slist;
    list<string> slist2(slist);

    // 容器末尾进行插入和删除
    push_back()
    pop_back()
    ```
    
- 泛型算法
    + <\algorithm>
    + 搜索
    + 排序
    + 复制、删除、替换
    + 关系
    + 生成
    + 数值
    + 集合
- 设计泛型算法
    + 
- Map
    
    ```cpp
    #include <map>
    #include <string>

    map<string, int> words;

    string tword;
    while (cin >> tword)
        words[tword]++;

    map<string, int>::iterator it = words.begin();
    for(; it != words.end(); ++it)
        cout << "key: " << it->first << "value: " << it->second << endl;

    // 查找key
    int count = 0;
    map<string, int>::iterator it;

    it = words.find("cer");
    if (it != words.end())
        count = it->second;

    int count = 0;
    string serach_word("cer");

    if (words.count(serach_word))
        count = words[serach_word];
    ```

- Set
    + 想知道某值是否存在于集合中，就可以使用set
- Iterator Inserter
    + 
- iostream Iterator
    + 

## 练习code
https://github.com/bjmsong/essentialCpp


## 参考资料
https://github.com/AndyHsu-cn/Essential_CPP