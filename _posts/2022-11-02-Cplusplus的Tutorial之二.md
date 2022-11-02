---
layout:     post
title:      Cplusplus的Tutorial
subtitle:   之二
date:       2022-11-02
author:     bjmsong
header-img: 
catalog: true
tags:
    - C++
---
>https://cplusplus.com/doc/tutorial/


## Compound data types
13. Dynamic Memory
- 在某些情况下，程序的内存需求只能在运行时确定。例如，当所需的内存取决于用户输入时。在这些情况下，程序需要动态分配内 存，为此c++语言集成了操作符new和delete。

```cpp
int * foo;
foo = new int [5];  // 为5个int类型的元素分配空间，返回一个指向序列第一个元素的指针，赋给foo
```

- 声明普通数组与使用new分配动态内存有很大的区别。最重要的区别是：常规数组的大小是一个常量表达式,它的大小必须在设计程序,运行程序之前确定,而动态内存分配允许在运行时分配任意内存大小。
- 程序所请求的动态内存是由系统从内存堆(heap)中分配的。然而，计算机内存可能会被耗尽。因此，不能保证所有使用operator new分配内存的请求都将被成功分配。
- C++提供了两种标准机制来检查分配是否成功:
    1. new默认使用的方法，通过处理异常，当分配失败时抛出bad_alloc类型的异常。如果抛出了这个异常，并且没有由特定的处理程序处理它，则程序执行将终止。
    2. nothrow，当内存分配失败时，new返回的指针是一个空指针，程序继续正常执行。

    ```cpp    
    int * foo;
    foo = new (nothrow) int [5]; 
    if (foo == nullptr) {
    // error assigning memory. Take measures.
    }
    ```

- 在大多数情况下，动态分配的内存只需要在程序中的特定时间段内使用;一旦不再需要它，就可以释放它，这样内存就可以再次用于其他动态内存请求。

    ```cpp
    delete pointer;   // 释放使用new分配的单个元素的内存
    delete[] pointer;  // 释放使用new和括号内的size([])分配的元素数组的内存
    ```

- 作为参数传递给delete的值必须是一个指针，该指针指向先前用new分配的内存块，或者是一个空指针(对于空指 针，delete不起作用)。
- C语言中的动态内存
    - <stdlib.h>：malloc , calloc , realloc
    - 这些函数在c++中也可用，也可以用来分配和释放动态内存。但是，由这些函数分配的内存块不一定与new返回的内存块兼容，它们不应混合使用。  

14. Data structure

```cpp
struct type_name { 
    member_type1 member_name1; 
    member_type2 member_name2; 
    member_type3 member_name3; 
    .
    .
} object_names;
```

- 指向结构体的指针

```cpp
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

struct movies_t { 
    string title; 
    int year;
};

int main () {
    string mystr;
    movies_t amovie; 
    movies_t * pmovie;
    pmovie = &amovie;
    cout << "Enter title: ";
    // ->是一个解引用操作符，仅用于指向具有成员的对象的指针。该操作符用于直接从对象的地址访问对象的成员
    getline (cin, pmovie->title); // 等价于(*pmovie).title
    cout << "Enter year: ";
    getline (cin, mystr);
    (stringstream) mystr >> pmovie->year;
    cout << "\nYou have entered:\n";
    cout << pmovie->title;
    cout << " (" << pmovie->year << ")\n";
    return 0; 
}
```

- 结构体可以嵌套

15. Other data types
- Type aliases 
    - 用途
        - 将程序从所使用的底层类型抽象出来
            - 例如，通过使用int的别名来引用特定类型的形参，而不是直接使用int，它允许该类型在以后的版本中很 容易被long(或其他类型)替换，而不必更改使用它的每个实例。
        - 减少⻓类型名或容易混淆的类型名的⻓度
    - typedef 继承自C语言

    ```cpp
    typedef char C;
    typedef unsigned int WORD;
    typedef char * pChar;
    typedef char field [50];
    ```

    - using C++11支持

    ```cpp
    using C = char;
    using WORD = unsigned int; 
    using pChar = char *; 
    using field = char [50];
    ```

- union
    - 其中所有成员元素在内存中占据相同的物理空间。这种类型的大小是最大的成员元素之一
    - 每个成员都具有不同的数据类型。但由于它们都指向内存中的相同位置，因此修改其中一个成员将影响所有成员的值

    ```cpp
    union mytypes_t { 
        char c;
        int i;
        float f;
    } mytypes;
    ```

- enum 枚举

    ```cpp
    enum colors_t {black, blue, green, cyan, red, purple, yellow, white};

    colors_t mycolor;
    mycolor = blue;
    if (mycolor == green) mycolor = red;

    // 用enum声明的枚举类型的值可以隐式转换为整数类型
    // 如果没有指定，则与第一个可能值相等的整数值为0，与第二个可能 值相等的整数值为1，第三个可能值为2，以此类推......因此，在上面定义的数据类型colors_t中，黑色等于0，蓝色 等于1，绿色等于2，以此类推......
    ```

## Classes
16. Classes (I)
- Basic
    - 类是数据结构(data structure)的扩展概念: 与数据结构一样，类可以包含数据成员，但也可以包含作为成员的函数
    - 对象是类的实例化。就变量而言，类是类型，对象是变量
    - 类可以使用关键字class，struct，union来定义
        - 用关键字struct声明的类的成员在默认情况下具有公共访问权限，而用关键字class声明的类的成员 在默认情况下具有私有访问权限
    - 访问控制符
        - private：同一个类的成员函数或者友元函数可以访问
        - protected：同一个类的成员函数或者友元函数，或者派生类可以访问
        - public：在对象可⻅的任何地方都可以访问
    - 用class关键字声明的类的所有成员默认是private（struct默认是public）

    ```cpp
    #include <iostream>
    using namespace std;

    class Rectangle {
        int width, height;
    public:
        void set_values (int,int);
        int area() {return width*height;}
    };
    void Rectangle::set_values (int x, int y) { 
        width = x;
        height = y;
    }
    int main () {
        Rectangle rect;
        rect.set_values (3,4);
        cout << "area: " << rect.area(); 
        return 0;
    }
    ```

    - 完全在类内定义的成员函数，与只在类中包含成员函数的声明，然后在类外定义的成员函数，两者之间的唯一区别是， 在第一种情况下，编译器会自动将该函数视为内联成员函数，而在第二个函数中，它是一个普通的(非内联的)类成员函数。这不会导致行为上的差异，但会影响可能的编译器优化。

- 构造函数 Constructors
    - 每当创建类的新对象时，都会自动调用构造函数，从而允许类初始化成员变量或分配存储空间
    - 构造函数的声明类似于普通成员函数，但具有与类名相同的名称，且没有任何返回类型(甚至void)
    - 不能像调用普通成员函数那样显式调用构造函数。它们只在该类的新对象创建时执行一次

    ```cpp
    #include <iostream>
    using namespace std;

    class Rectangle {
        int width, height;
      public:
        Rectangle (int,int);
        int area () {return (width*height);}
    };

    Rectangle::Rectangle (int a, int b) { 
        width = a;
        height = b;
    }

    int main () {
        Rectangle rect (3,4);
        Rectangle rectb (5,6);
        cout << "rect area: " << rect.area() << endl; 
        cout << "rectb area: " << rectb.area() << endl; 
        return 0;
    }
    ```

- 重载构造函数
    - 与任何其他函数一样，构造函数也可以重载带有不同形参的不同版本:具有不同数量的形参和/或不同类型的形参。 编译器会自动调用形参与实参匹配的函数
    - 默认构造函数是不接受形参的构造函数
- Uniform initialization
    
    ```cpp
    #include <iostream>
    using namespace std;

    class Circle {
        double radius;
    public:
        Circle(double r) { radius = r; }
        double circum() {return 2*radius*3.14159265;}
    };
    int main () {
        Circle foo (10.0);   // functional form
        Circle bar = 20.0;   // assignment init.
        Circle baz {30.0};   // uniform init.
        Circle qux = {40.0}; // POD-like
        
        cout << "foo's circumference: " << foo.circum() << '\n';
        return 0; 
    }
    ```

- 构造函数中初始化成员变量
    - 构造函数可以直接初始化成员变量，不一定需要在构造函数体中来初始化

    ```cpp
    #include <iostream>
    using namespace std;

    class Circle {
        double radius;
    public:
        Circle(double r) : radius(r) { }
        double area() {return radius*radius*3.14159265;}
    };

    class Cylinder {
        Circle base;
        double height;
      public:
        Cylinder(double r, double h) : base (r), height(h) {}
    double volume() {return base.area() * height;} };
    
    int main () {
        Cylinder foo (10,20);
        cout << "foo's volume: " << foo.volume() << '\n';
        return 0; 
    }

    // 这些初始化也可以使用统一的初始化语法，使用大括号{}而不是圆括号()
    Cylinder::Cylinder (double r, double h) : base{r}, height{h} { }
    ```
- 指向类的指针

    ```cpp
    #include <iostream>
    using namespace std;

    class Rectangle {
      int width, height;
    public:
      Rectangle(int x, int y) : width(x), height(y) {}
      int area(void) { return width * height; }
    };

    int main() {
        Rectangle obj (3, 4);
        Rectangle * foo, * bar, * baz;
        foo = &obj;
        bar = new Rectangle (5, 6);
        baz = new Rectangle[2] { {2,5}, {3,6} };
        cout << "obj's area: " << obj.area() << '\n';
        cout << "*foo's area: " << foo->area() << '\n'; 
        cout << "*bar's area: " << bar->area() << '\n'; 
        cout << "baz[0]'s area:" << baz[0].area() << '\n'; 
        cout << "baz[1]'s area:" << baz[1].area() << '\n'; 
        delete bar;
        delete[] baz;
        return 0;
    }
    ```

17. Classes (II)
- 重载操作符
    - C++允许重载大多数操作符

    ```cpp
    #include <iostream>
    using namespace std;

    class CVector {
      public:
        int x,y;
        CVector () {};
        CVector (int a,int b) : x(a), y(b) {}
        CVector operator + (const CVector&);
    };

    CVector CVector::operator+ (const CVector& param) {
        CVector temp;
        temp.x = x + param.x;
        temp.y = y + param.y;
        return temp;
    }

    int main () {
        CVector foo (3,1);
        CVector bar (1,2);
        CVector result;
        result = foo + bar;
        cout << result.x << ',' << result.y << '\n'; 
        return 0;
        }
    ```

- this关键字
    - 表示一个指向其成员函数正在执行的对象的指针，在类的成员函数中使用它来引用对象本身。

    ```cpp
    #include <iostream>
    using namespace std;

    class Dummy {
      public:
        bool isitme (Dummy& param);
    };
    bool Dummy::isitme (Dummy& param)
    {
        if (&param == this) 
            return true;
        else 
            return false;
    }

    int main () {
        Dummy a;
        Dummy* b = &a;
        if ( b->isitme(a) )
            cout << "yes, &a is b\n"; 
        return 0;
    }
    ```

- 静态成员（数据或函数）
    - 一个类的静态数据成员也被称为“类变量”，因为同一个类的所有对象只有一个公共变量
    - 为了避免多次声明它们，不能在类 中直接初始化它们，而是需要在类外部进行初始化
- Const member functions
    - 当类的对象限定为const对象时，从类外部访问其数据成员的权限被限制为只读，就好像从类外部访问其数据成员的所有成员都是const一样。但是请注意，构造函数仍然被调用，并且允许初始化和修改这些数据成员。

    ```cpp
    #include <iostream>
    using namespace std;

    class MyClass {
      public:
        int x;
        MyClass(int val) : x(val) {}
        int get() {return x;}
    };

    int main() {
        const MyClass foo(10);
        foo.x = 20; // not valid: x cannot be modified 
        cout << foo.x << '\n'; // ok: data member x can be read 
        return 0;
    }
    ```
    - const对象的成员函数只有在自身被指定为const成员时才能被调用;在上面的例子中，成员get(没有指定为const)不 能从foo中调用
    - 要指定一个成员为const成员，const关键字必须紧跟在函数原型的形参的右括号之后
    - 指定为const的成员函数不能修改非静态数据成员，也不能调用其他非const成员函数。本质上，const成员不能修 改对象的状态

- 类模板

    ```cpp 
    template <class T>
    class mypair {
        T values [2];
      public:
        mypair (T first, T second)
        {
            values[0]=first; 
            values[1]=second;
        }
    };

    mypair<int> myobject (115, 36);
    ```

- Template specialization

18. Special members 特殊成员函数
- 特殊成员函数是隐式定义的成员函数。有六个:
|Member function |typical form for class C :
|  ----  | ----  |
| Default constructor | C::C(); |
| Destructor | C::~C(); |
| Copy constructor | C::C (const C&); |
| Copy assignment | C& operator= (const C&); |
| Move constructor | C::C (C&&); |
| Move assignment | C& operator= (C&&); |
~
- 默认构造函数
    - 默认构造函数是在声明类的对象时调用的构造函数，但不使用任何参数初始化。
- 构造函数
    - 
- 拷贝构造函数
    - 
- 拷贝赋值
    - 
- 移动构造函数、移动赋值
    - 
- 

19. Friendship and inheritance
- 友元
- 继承

20. Polymorphism 多态
- 
- 


## Other language features
21. Type conversions
- 
- explicit


22. Exceptions
- 异常
- 

23. Preprocessor directives 预处理器
- define
- include

## Standard library
24. Input/output with files
- 
- 


## 参考资料
- 24节课快速过完c++所有语法
https://www.bilibili.com/video/BV1Gv411N7KN/?spm_id_from=333.999.0.0&vd_source=7798c62f92ce545f56fd00d4daf55e26
