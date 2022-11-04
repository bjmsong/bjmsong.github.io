---
layout:     post
title:      Cplusplus的Tutorial
subtitle:   之一
date:       2022-10-29
author:     bjmsong
header-img: 
catalog: true
tags:
    - C++
---
>https://cplusplus.com/doc/tutorial/

## Basic of C++
1. Structure of a program
- namespace

2. Variables and types
- 基本数据类型
    - Character，Numerical integer，Floating-point，Boolean
    - 可以简写：例如signed short int  可以简写为short
    - 每一种基本数据类型都有多个类型(例如char，char16_t，char32_t)，唯一的区别是占的内存空不一样
    - 除了char，其它数据类型占的内存大小都不确定，由编译器和机器来决定
    - <\limits.h>: 决定了各种变量类型的各种属性
    - <\cstdint.h>: 定义了一系列特定长度的类型别名
- void
- nullptr
- 变量要先声明，再使用
- 三种初始化方式

    ```cpp
    int x = 0;
    int x (0);
    int x {0};  // C++11支持
    ```
    
- 类型(自动)推断

    ```cpp
    int foo = 0;
    auto bar = foo;   // C++11支持

    int foo = 0;
    decltype(foo) bar; 
    ```

- string

3. Constants 常量
- Literals 字面量
    - 文字常量可以分为:整型、浮点型、字符型、字符串型、布尔型、指针型和用户定义的文字型。
- constant
- #define：发生在编译之前 

4. Operators
- 赋值: 从右到左，可以连续赋值，赋值本身有一个值
- 算术运算符
- 复合赋值
- 递增，递减
- 关系运算符
- 逻辑运算符
- 条件运算符
- 位运算符
- 类型强制转换

    ```cpp
    int i;
    float f = 3.14; 
    i = (int) f;    // C语言继承来的写法
    i = int (f);
    ```

- sizeof
    - 接受一个形参，形参可以是类型或变量，并返回该类型或对象的大小(以字节为单位)

5. Basic Input/Output
- cin,cout
- getline
- stringstream: 将字符串转换为数值


## Program Structure
6. Statements and flow control
- if-else
- for
    - for (initialization; condition; increase) statement;
    - for ( declaration : range ) statement; C++11支持

    ```cpp
    // range-based for loop
    #include <iostream>
    #include <string>
    using namespace std;
    int main () {
        string str {"Hello!"}; 
        for (char c : str)
        {
        cout << "[" << c << "]"; }
        cout << '\n'; 
    }
    ```

- while
- break, continue
- 用的比较少
    - switch
    - goto
    - do while

7. Functions
- 传递参数
    - 按值(by value)传递：将实参的副本传递给被调用函数，函数不会影响到实参本身
    - 按引用(by reference)传递: 函数将其参数声明为引用，传递的是实参本身，函数会修改实参

    ```cpp
    #include <iostream>
    using namespace std;
    void duplicate (int& a, int& b, int& c)
    {
        a*=2; b*=2; c*=2;
    }
    int main () {
        int x=1, y=3, z=7;
        duplicate (x, y, z);
        cout << "x=" << x << ", y=" << y << ", z=" << z; 
        return 0;
    }
  ```

    - 如果参数是大型复合类型，按值传递的开销较大，按引用传递可以避免这种开销。如果同时想要参数不被修改，可以奖赏const修饰符

    ```cpp
    string concatenate (const string& a, const string& b)
    {
        return a+b; 
    }
    ```
- 内联函数 inline
    - 调用函数通常会引起一定的开销(堆叠参数、跳转等)，因此对于非常短的函数，在调用函数的地方简单地插入函数 的代码可能更有效，而不是执行正式调用函数的过程。
    - 在函数声明之前加上inline说明符，告知编译器对于该函数，内联展开优于通常的函数调用机制。这不会 改变函数的行为，只是用来建议编译器在调用函数的每个点插入函数体生成的代码，而不是使用常规的函数调用。
    
    ```cpp 
    inline string concatenate (const string& a, const string& b)
    {
        return a+b; 
    }
    ```

    - 大多数编译器在发现有机会提高效率时，即使没有显式地用内联说明符标记，也已经优化代码以生成内联 函数。因此，这个说明符仅仅指示编译器，内联是该函数的首选，尽管编译器可以自由地不内联它，或者优化它。 在c++中，优化是委托给编译器的任务，只要结果行为是代码指定的行为，编译器就可以自由地生成任何代码。
- 先声明，再调用
- 递归

8. Overloads and templates
- 重载

    ```cpp
    #include <iostream>
    using namespace std;
    int operate (int a, int b)
    {
      return (a*b);
    }
    double operate (double a, double b)
    {
      return (a/b);
    }

    int main () {
        int x=5,y=2;
        double n=5.0,m=2.0;
        cout << operate (x,y) << '\n'; 
        cout << operate (n,m) << '\n'; 
        return 0;
    }
    ```

- 模板：使用泛型类型定义函数
    - 模板形参可以有多个，可以包含泛型，也可以包含指定类型

```cpp
#include <iostream>
using namespace std;
template <class T>
T sum (T a, T b)
{
T result; result = a + b; return result;
}

int main () {
    int i=5, j=6, k;
    double f=2.0, g=0.5, h; 
    k=sum<int>(i,j); 
    h=sum<double>(f,g);
    cout << k << '\n'; cout << h << '\n'; 
    return 0;
}

```

9. Name visibility
- 全局变量，局部变量
- Namespaces 命名空间
    - using关键字将名称引入到当前声明区域(如块)，从而避免了对名称进行限定的需要


## Compound data types
10. Arrays
- 内置数组
    - 不同的初始化方式

    ```cpp
    int foo [5] = { 16, 2, 77, 40, 12071 };
    int bar [5] = { 10, 20, 30 };
    int baz [5] = { };
    int foo [] = { 16, 2, 77, 40, 12071 };
    int foo[] { 10, 20, 30 };
    ```

    - 多维数组
- 为了克服语言内置数组的一些问题，c++提供了另一种数组类型作为标准容器，定义在头文件<array.h>

```cpp
// 使用内置数组
#include <iostream>
using namespace std;

int main() {
    int myarray[3] = {10,20,30};
 
    for (int i=0; i<3; ++i) 
        ++myarray[i];
    for (int elem : myarray) 
        cout << elem << '\n';
}

// 使用标准库中的容器
 
#include <iostream>
#include <array>
using namespace std;
int main() {
    array<int,3> myarray {10,20,30};
    for (int i=0; i<myarray.size(); ++i)
        ++myarray[i];
    for (int elem : myarray) 
        cout << elem << '\n';
}
``` 

11. Character sequences
- 字符串可以用string类处理，也可以将它们表示为字符类型元素的普通数组(c字符串)

```cpp
  char foo [20];

  char myword[] = { 'H', 'e', 'l', 'l', 'o', '\0' }; 
  char myword[] = "Hello";
```

    - 用字符序列表示的字符串以'\0'结尾
- 在标准库中，字符串的两种表示形式(c字符串和标准库字符串)同时存在
    - 以空结尾的字符序列可以隐式转换为字符串，而字符串可以通过使用string的成员函数c_str或data转换为以空结尾
的字符序列

```cpp
char myntcs[] = "some text";
string mystring = myntcs; // convert c-string to string 
cout << mystring; // printed as a library string 
cout << mystring.c_str(); // printed as a c-string
```

12. Pointers
- 存储另一个变量地址的变量称为指针

```cpp
#include <iostream>
using namespace std;
int main () {
    int firstvalue, secondvalue;
    int * mypointer;
    mypointer = &firstvalue;  // 取址操作符
    *mypointer = 10;          // 解引用操作符
    mypointer = &secondvalue;
    *mypointer = 20;
    cout << "firstvalue is " << firstvalue << '\n'; 
    cout << "secondvalue is " << secondvalue << '\n'; return 0;
}

```

- 指针和数组
    - 数组的名称代表数组第一个元素的内存地址
    - 指针和数组支持相同的操作，两者的含义相同。主要的区别在于指针可以被分配新的地址，而数组则不能。
    - 括号([])可以被解释为指定数组元素的下标。实际上，这些括号是一个解引用操作符，称为偏移操作符。它们会像*一样解除对后面的变量的引用，但它们也会将括号之间的数字加到要解除引用的地址上

    ```cpp
    // 这两个表达式等价且有效，不仅当a是指针时有效，而且当a是数组时也有效。记住，如果一个数组，它的名称可以 像指向它的第一个元素的指针一样使用。
     a[5] = 0; // a [offset of 5] = 0 
     *(a+5) = 0; // pointed to by (a+5) = 0
    ```
- 指针初始化
    
    ```cpp
    // 指针可以初始化为变量的地址，也可以初始化为另一个指针(或数组)的值
    int myvar;
    int *foo = &myvar;
    int *bar = foo;
    ```

- 指针算术
    - 只允许加减
    - 当给指针加1时，指针将指向同一类型的下一个元素，因此，指针所指向的类型的字节大小将被添加到指针上。
    - 自增/自减操作符可以用作表达式的前缀或后缀，但行为略有不同:作为前缀，increment发生在表达式求值之前，作为后缀，increment发生在表达式求值之后。

    ```cpp
    *p++ // same as *(p++): increment pointer, and dereference unincremented address 自增 指针和解引用未自增地址
    *++p // same as *(++p): increment pointer, and dereference incremented address 自增指针 和解引用自增地址
    ++*p // same as ++(*p): dereference pointer, and increment the value it points to 解引 用指针，并使其指向的值递增
    (*p)++ // dereference pointer, and post-increment the value it points to 解引用指针，并对 指针所指向的值进行后加
    ```

- 指针和常量
    
    ```cpp 
    int x;
    int y = 10;
    const int * p = &y;    // 指针指向的对象的值不能改变，可以修改指针指向的对象。也可以写成 int const * p
    x = *p; // ok: reading p
    *p = x; // error: modifying p, which is const-qualified
    ```

    ```cpp
    // 指向const元素的指针的一个用例是作为函数形参:接受非const指针作为形参的函数可以修改作为实参传递的值，而 接受const指针作为形参的函数则不能。
    #include <iostream>
    using namespace std;

    void increment_all (int* start, int* stop) {
        int * current = start; 
        while (current != stop) {
            ++(*current);  // increment value pointed
            ++current;     // increment pointer
            } 
    }

    void print_all (const int* start, const int* stop) {
        const int * current = start; 
        while (current != stop) {
        cout << *current << '\n';
            ++current;     // increment pointer
          }
    }
    int main () {
        int numbers[] = {10,20,30}; 
        increment_all (numbers,numbers+3); 
        print_all (numbers,numbers+3); 
        return 0;
    }
    ```

    ```cpp
    int x;
    int * p1 = &x; // non-const pointer to non-const int
    const int * p2 = &x; // non-const pointer to const int 
    int * const p3 = &x; // const pointer to non-const int
    const int * const p4 = &x; // const pointer to const int
    ```

- 指针和字符串字面量

    ```cpp
    const char * foo = "hello";
    // 访问元素
    *(foo+4)
    foo[4]
    ```

- 指向指针的指针

    ```cpp
    char a; 
    char * b; 
    char ** c; 
    a = 'z';
    b = &a;
    c = &b;
    ```

- void类型指针
    - 能够指向任何数据类型，但是指向的数据不能直接引用，需要被转换成其他类型指针之后才能被引用

    ```cpp
    // 用途之一是向函数传递泛型参数
    #include <iostream>
    using namespace std;

    void increase (void* data, int psize)
    {
        if ( psize == sizeof(char) )
        { char* pchar; pchar=(char*)data; ++(*pchar); } 
        else if (psize == sizeof(int) )
        { int* pint; pint=(int*)data; ++(*pint); }
    }
    int main () {
        char a = 'x';
        int b = 1602;
        increase (&a,sizeof(a)); 
        increase (&b,sizeof(b));
        cout << a << ", " << b << '\n'; 
        return 0;
    }
    ```

- 无效指针和空指针

    ```cpp
    // 无效指针：例如未初始化的指针和指向数组中不存在元素的指针 
    int * p;               // uninitialized pointer (local variable)
    int myarray[10];
    int * q = myarray+20; // element out of bounds

    // 上面的语句都不会导致错误。在c++中，指针被允许接受任何地址值，不管这个地址上是否真的有东⻄。导致错误的是对这样的指针进行解引用。

    // 任何指针类型都可以接受的特殊值:空指针值
    int * p = 0;
    int * q = nullptr;

    // 不要混淆null pointers 和 void pointers
    ```

- 指向函数的指针

    ```cpp
    #include <iostream>
    using namespace std;

    int addition (int a, int b)
    { return (a+b); }
    int subtraction (int a, int b)
    { return (a-b); }
    int operation (int x, int y, int (*functocall)(int,int))
    {
    int g;
    g = (*functocall)(x,y); return (g);
    }
    int main () {
        int m,n;
        int (*minus)(int,int) = subtraction;
        m = operation (7, 5, addition); 
        n = operation (20, m, minus); 
        cout <<n;
        return 0;
    }
    ```

## 参考资料
- 24节课快速过完c++所有语法
https://www.bilibili.com/video/BV1Gv411N7KN/?spm_id_from=333.999.0.0&vd_source=7798c62f92ce545f56fd00d4daf55e26
