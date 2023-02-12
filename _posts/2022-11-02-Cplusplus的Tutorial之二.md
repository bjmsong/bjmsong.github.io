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
    - <\stdlib.h>：malloc , calloc , realloc
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
        - private：同一个类的成员函数，友元函数可以访问
        - protected：同一个类的成员函数，友元函数，派生类可以访问
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
    - 为了避免多次声明它们，不能在类中直接初始化它们，而是需要在类外部进行初始化
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
|Member function |typical form for class C :|
|  ----  | ----  |
| Default constructor | C::C(); |
| Destructor | C::\~C(); |
| Copy constructor | C::C (const C&); |
| Copy assignment | C& operator= (const C&); |
| Move constructor | C::C (C&&); |
| Move assignment | C& operator= (C&&); |

- 默认构造函数 Default constructor
    - 默认构造函数是在声明类的对象时调用的构造函数，但不使用任何参数初始化
    - 如果类定义没有构造函数，编译器就假定该类具有隐式定义的默认构造函数，类的对象可以通过简单地声明而不带任何参数来构造
    - 一旦类的某个构造函数显式声明了任意数量的形参，编译器就不再提供隐式默认构造函数，也不再允许声明 没有实参的类的新对象。如果需要在不带参数的情况下构造该类的对象，也应该在类中声明适当的默认构造函数。
- 析构函数 Destructor
    - 析构函数实现构造函数相反的功能: 它们负责在类生命周期结束时进行必要的清理
    
    ```cpp
    #include <iostream>
    #include <string>
    using namespace std;

    class Example4 {
        string* ptr;
    public:
    // constructors:
    Example4() : ptr(new string) {}
    Example4 (const string& str) : ptr(new string(str)) {} // destructor:
    ~Example4 () {delete ptr;}
    // access content:
    const string& content() const {return *ptr;}
    };

    int main () {
        Example4 foo;
        Example4 bar ("Example");
        cout << "bar's content: " << bar.content() << '\n';
        return 0; 
    }
    ```

- 拷贝构造函数 Copy constructor
    - 当一个对象被传递一个相同类型的对象作为参数时，拷贝构造函数被调用来创建一个副本
    - 拷贝构造函数的第一个形参是类本身的引用
    - 如果类没有定义自定义复制或移动构造函数(或赋值)，则提供隐式复制构造函数，这个函数的定义执行了一个浅拷⻉

    ```cpp
    MyClass::MyClass(const MyClass& x) : a(x.a), b(x.b), c(x.c) {}
    ```

    ```cpp
    // copy constructor: deep copy
    // 深拷贝会为新的string开辟一块内存区域，这块区域会被初始化为原始对象的副本。
    #include <iostream>
    #include <string>
    using namespace std;

    class Example5 {
        string* ptr;
        public:
            Example5 (const string& str) : ptr(new string(str)) {} ~Example5 () {delete ptr;}
            // copy constructor:
            Example5 (const Example5& x) : ptr(new string(x.content())) {} // access content:
            const string& content() const {return *ptr;}
    };

    int main () {
        Example5 foo ("Example"); 
        Example5 bar = foo;
        cout << "bar's content: " << bar.content() << '\n';
        return 0; 
    }
    ```

- 拷贝赋值 Copy assignment
    - 对象不仅可以在构造时被复制，也可以在任何的赋值操作时被复制
    - 就是对"="运算符的重载

    ```cpp
    MyClass foo; 
    MyClass bar (foo);   // object initialization: copy constructor called
    MyClass baz = foo;   // object initialization: copy constructor called
    foo = bar;           // object already initialized: copy assignment called
    ```

- 移动构造函数、移动赋值 Move constructor and assignment
    - 与拷贝构造不同的是，内容从一个对象移动到了另一个对象，源对象丢失了内容，只有当源是一个未命名对象时才可以
    - 未命名对象的典型例子是函数的返回值、类型转换

    ```cpp
    MyClass fn();   // function returning a MyClass object
    MyClass foo;    // default constructor
    MyClass bar = foo;   // copy constructor
    MyClass baz = fn();   // move constructor
    foo = bar;    // copy assignment
    baz = MyClass();  // move assignment
    ```

    - 移动构造函数和移动赋值的形参是类的右值引用(rvalue reference)，右值引用很少用于移动构造函数之外的场景

    ```cpp
    MyClass (MyClass&&); // move-constructor 
    MyClass& operator= (MyClass&&); // move-assignment
    ```

- Implicit members 隐式成员
    - 每个类都可以显式地选择哪些成员具有默认定义，或者分别使用关键字default和delete删除哪些成员。语法是:

    ```cpp
    #include <iostream>
    using namespace std;

    class Rectangle {
        int width, height;
    public:
        Rectangle (int x, int y) : width(x), height(y) {} 
        Rectangle() = default;
        Rectangle (const Rectangle& other) = delete;
        int area() {return width*height;}
    };

    int main () {
        Rectangle foo;
        Rectangle bar (10,20);
        cout << "bar's area: " << bar.area() << '\n';
        return 0; 
    }
    ```

19. Friendship and inheritance
- 友元
    - 友元函数
        - 可以访问类的private和protected成员
        - 不是类的成员函数

    ```cpp
    class Rectangle {
        int width, height;
      public:
        Rectangle() {}
        Rectangle (int x, int y) : width(x), height(y) {}
        int area() {return width * height;}
        friend Rectangle duplicate (const Rectangle&);
    };
    Rectangle duplicate (const Rectangle& param)
    {
        Rectangle res;
        res.width = param.width*2; 
        res.height = param.height*2; 
        return res;
    }

    int main () {
        Rectangle foo;
        Rectangle bar (2,3);
        foo = duplicate (bar);
        cout << foo.area() << '\n'; 
        return 0;
    }
    ```

    - 友元类
        - 其成员可以访问另一个类的private或protected成员
- 继承
    - public派生类继承基类的所有可访问成员，除了
        - constructors and its destructor
        - assignment operator members (operator=)
        - friends
        - private members
    - 派生类的成员可以访问基类的受保护成员，但不能访问它的私有成员
    - 多继承
        - 从多个类继承

    ```cpp
    class Polygon {
      protected:
        int width, height;
      public:
        void set_values (int a, int b) 
            { width=a; height=b;}
    };

    class Rectangle: public Polygon {
      public:
        int area ()
          { return width * height; }
    };

    int main () {
        Rectangle rect;
        Triangle trgl; 
        rect.set_values (4,5); 
        trgl.set_values (4,5);
        cout << rect.area() << '\n'; 
        cout << trgl.area() << '\n'; 
    }
    ```

20. Polymorphism 多态
- 类继承的一个关键特性是，指向派生类的指针与指向基类的指针类型兼容。多态性是利用这个简单但强大和通用特性的艺术。

```cpp
class Polygon {
  protected:
    int width, height;
  public:
    void set_values (int a, int b) 
        { width=a; height=b; }
};

class Rectangle: public Polygon {
  public:
    int area()
      { return width*height; }
};

class Triangle: public Polygon {
  public:
    int area()
      { return width*height/2; }
};

int main () {
    Rectangle rect;
    Triangle trgl;
    Polygon * ppoly1 = &rect; 
    Polygon * ppoly2 = &trgl; 
    ppoly1->set_values (4,5); 
    ppoly2->set_values (4,5); 
    cout << rect.area() << '\n'; 
    cout << trgl.area() << '\n';
}
```
- 虚函数
    - 可以在派生类中重新定义的成员函数，同时通过引用保留其调用属性。要变成虚函数的语法是在它的声明之前加上virtual关键字
    - 实现多态的基石
        - 子类重写父类的虚函数
        - 基类指针指向派生类，基类指针调用虚函数时会调用派生类的虚函数
    - 原理
        - 编译器会给含有virtual function的类加上一个成员变量---虚函数指针(vptr)。vptr指向虚函数表(vftable)，虚函数表存储指向（对应）虚函数的地址
        - non-virtual function在编译期被resolve，virtual function在runtime被resolve。
        - https://www.bilibili.com/video/BV1R7411p79b/?spm_id_from=333.999.0.0&vd_source=7798c62f92ce545f56fd00d4daf55e26
- 抽象基类是只能用作基类的类，允许具有没有定义的虚成员函数(称为纯虚函数)。语法是将它们的定义替换为=0(一个等号和一个0)。抽象基类不能用于实例化对象


## Other language features
21. Type conversions
- Implicit conversion
    - 当值被复制到兼容类型时，隐式转换将自动执行
    
    ```cpp
    short a=2000; 
    int b;
    b=a;
    ```

    - 对于非基本类型，数组和函数隐式转换为指针，指针通常允许以下转换:
        + null pointer可以转换为任何类型的指针
        + 指向任何类型的指针都可以转换为void pointer
        + 指向派生类的指针可以转换为可访问且无二义性的基类的指针
- Implicit conversions with classes
    + 在类的世界里，隐式转换可以通过三个成员函数来控制:
        * 单参数构造函数:允许从特定类型隐式转换来初始化对象
        * 赋值操作符:允许在赋值时从特定类型隐式转换
        * 类型强制转换操作符:允许隐式转换到特定类型
- keyword explicit
- Type casting 强制类型转换
    + 为了控制类之间的这些转换类型，我们有四个特定的强制转换操作符:dynamic_cast、reinterpret_cast、 static_cast和const_cast。它们的格式是紧跟在尖括号(<>)之间的新类型之后，紧接在要在括号之间转换的表达式之后。
    
    ```cpp
    dynamic_cast <new_type> (expression)
    reinterpret_cast <new_type> (expression)
    static_cast <new_type> (expression)
    const_cast <new_type> (expression
    ```
- https://www.cnblogs.com/fortunely/p/14453815.html
- static_cast
    - 任何具有明确定义的类型转换，只要不包含底层const，都可以使用static_cast。
- dynamic_cast
    + 一般情况下，不允许将基类指针或引用转换成派生类指针或引用。dynamic_cast让类型转换的安全检查在运行时执行，从而安全地将基类指针或引用，转换为派生类指针或引用。
- const_cast
    - 去掉底层const
- reinterpret_cast
    + 
- typeid
- RTTI


22. Exceptions
- 异常
    - 异常提供了一种意外情况的处理方法，将控制权转移到特殊的函数(handler)
    - 为了捕获异常，通过将代码的一部分封装在try块中，可以把它们置于异常检查之下。当该块中出现 异常情况时，将抛出一个异常，将控制权传递给异常处理程序。如果没有抛出异常，则代码继续正常运行，并忽略 所有处理程序。
    - 在try块内部使用throw关键字会抛出异常。异常处理程序是用关键字catch声明的，该关键字必须立即放在try块之后。

    ```cpp
    int main () {
      try
      {
        throw 20;
      }
      catch (int e)  // 形参只有跟throw表达式传递的实参的类型匹配的情况下，才会捕获异常
      {
        cout << "An exception occurred. Exception Nr. " << e << '\n'; 
      }
        return 0;
    }
    ```

    - 如果使用省略号(...)作为catch的形参，则该处理程序将捕获任何异常，而不管抛出的异常类型是什么。这可以用作 一个默认的处理程序，用来捕获其他处理程序没有捕获的所有异常
- 标准异常
    - C++标准库提供了一个基类，专⻔用于声明作为异常抛出的对象。它被称为std::exception，定义在头文件<\exception>中。这个类有一个名为what的虚成员函数，它返回一个以空字符结尾的字符序列(类型为char \*)，并且可以在派生类中重写 该函数以包含异常的某种描述。

    ```cpp
    #include <iostream>
    #include <exception>
    using namespace std;
    
    class myexception: public exception
    {
      virtual const char* what() const throw()
      {
    return "My exception happened"; }
    } myex;

    int main () {
         try
          {
            throw myex;
          }
          catch (exception& e)
          {
            cout << e.what() << '\n'; 
          }
        return 0;
        }
    ```

    - c++标准库组件抛出的所有异常都抛出派生自这个异常类的异常。
    - 头文件<\exception>定义了两种通用异常类型，派生自异常类exception，可被自定义异常继承以报告错误：

    |exception| description|
    |  ----  | ----  |
    | logic_error | error related to the internal logic of the program |
    | runtime_error | error detected during runtime |

23. Preprocessor directives 预处理器
- 预处理器指令是包含在程序代码中前面有#号(#)的行。这些行不是程序语句，而是预处理器的指令。预处理器在实 际编译代码之前检查代码，并在常规语句实际生成代码之前解析所有这些指令
- 宏定义 macro definitions
    - 替换代码其余部分中出现的任何标识符，这个替换可以是表达式、语句、块或任何东⻄

    ```cpp
    #define TABLE_SIZE 100
    int table1[TABLE_SIZE]; 
    #undef TABLE_SIZE     // 取消宏定义
    #define TABLE_SIZE 200 
    int table2[TABLE_SIZE];
    ```

- 条件引入 Conditional inclusions
    - 允许在满足特定条件时包含或丢弃程序的部分代码
    - #ifdef, #ifndef, #if, #endif, #else and #elif
- 行控制 Line control
    - 当我们编译一个程序时，在编译过程中发生了一些错误，编译器会显示一个错误消息，其中引用了发生错误的文件 的名称和行号，因此更容易找到生成错误的代码
    - #line指令允许我们控制这两件事：代码文件中的行号，以及当错误发生时我们希望出现的文件名
- Error directive
    - 当找到它时，这个指令会终止编译过程，生成一个编译错误，可以指定为它的参数
    - #error
- Source file inclusion
    
    ```cpp
    #include <header> 
    #include "file"
    ```
- Pragma directive
    - 用于向编译器指定不同的选项
- Predefined(预定义) macro names
    - 都以' __ '开始和结束

## Standard library
24. Input/output with files
- C++提供了以下类来执行文件的字符输出和输入:
    - ofstream : Stream class to write on files
    - ifstream : Stream class to read from files
    - fstream : Stream class to both read and write from/to files
    - 这些类直接或间接地派生自istream和ostream类。我们已经使用了这些类类型的对象:cin是istream类的对象， cout是ostream类的对象
    - 我们可以像使用cin和cout一样使用文件流，唯一的区别是我们必须将这些流与物理文件关联起来

    ```cpp
    #include <iostream>
    #include <fstream>
    #include <string>
    using namespace std;

    int main () {
        ofstream myfile ("example.txt"); 
        if (myfile.is_open())
        {
            myfile << "This is a line.\n"; 
            myfile << "This is another line.\n"; 
            myfile.close();
        }
        else cout << "Unable to open file"; 

        string line;
        ifstream myfile ("example.txt"); 
        if (myfile.is_open())
        {
            while ( getline (myfile,line) )
            {
                cout << line << '\n'; 
            }
            myfile.close(); 
        }
        else cout << "Unable to open file";
    }
    ```
- 以下成员函数用于检查流的特定状态(它们都返回bool值):
    - bad(): 如果读或写操作失败，则返回true
    - fail(): 在与bad()相同的情况下返回true，但在发生格式错误的情况下也返回true
    - eof(): 如果打开读取的文件已经到达末尾，则返回true
    - good(): 在调用前面的任何函数都会返回true的情况下，它会返回false
- 二进制文件
    - 对于二进制文件，使用提取和插入操作符(<<和>>)以及getline这样的函数读写数据是没有效率的，因为我们不需要 格式化任何数据，而且数据很可能不会在行中进行格式化
    - 文件流包括两个专⻔用于顺序读写二进制数据的成员函数:写入和读取。第一个(write)是ostream的成员函数(由
    ofstream继承)。read是istream的成员函数(由ifstream继承)
- Buffers and Synchronization
    - 当我们操作文件流时，它们与类型为streambuf的内部缓冲区对象相关联。这个缓冲区对象可能代表一个充当流和 物理文件之间的中介的内存块。例如，对于ofstream，每次调用成员函数put(写入单个字符)时，字符可能会被插 入这个中间缓冲区，而不是直接写入与流相关联的物理文件。

## 参考资料
- 24节课快速过完c++所有语法
https://www.bilibili.com/video/BV1Gv411N7KN/?spm_id_from=333.999.0.0&vd_source=7798c62f92ce545f56fd00d4daf55e26
