---
layout:     post
title:      effective python之协作开发
subtitle:   
date:       2020-02-12
author:     bjmsong
header-img: img/language/python.jpg
catalog: true
tags:
    - python
---

> Python的某些特性，可以帮助开发者构建接口清晰、边界明确的优秀API。Python开发者之间也形成了一套固定的做法，能够在程序的演化过程中尽量保持代码的可维护性。



### 49. 为每个函数、类和模块编写文档字符串

- Python是一门动态语言，所以文档极其重要，Python将文档视为第一等级的对象

- Python程序在运行的时候，能够直接访问源代码中的文档信息

  - 通过`__doc_`属性

  ```
  print(repr(functionname.__doc__))
  ```

  - 通过内置的help函数

- Python开发者社区构建了一些工具，可以把纯文本转换成HTML等更友好的格式

  - Sphinx
  - Read the Docs

- docstring使用三重双引号（"""）把它括起来，要及时更新，通过内置的doctest模块，可以运行docstring中的范例代码，以确保源代码和文档不会产生偏差

- 为模块编写文档

  - 描述本模块的用途
  - 介绍本模块的操作有关的内容
  - 强调本模块里面比较重要的类和函数
  
- 为类编写文档

  - 描述本类的用途
  - 介绍该类的操作方式
  - 介绍public属性及方法
  - 介绍如何与protected属性，超类方法交互
  
- 为函数编写文档

  - 描述本函数的功能
  - 描述具体的行为和函数的参数，返回值
  - 描述可能抛出的异常



### 50. 用包来安排模块，并提供稳固的API

- [廖雪峰-模块](https://www.liaoxuefeng.com/wiki/1016959663602400/1017454145014176)
- 模块：一个.py文件，module_name就是这个文件去掉.py之后的文件名，文件中可以直接定义一些变量、函数、类
- 程序的代码量变大之后，需要重新调整结构
  - 大函数分割成小函数
  - 把某些数据结构重构为辅助类
  - 把功能分散到多个相互依赖的模块之中
- 模块太多，引入一种抽象层，使得代码更易于理解 -- 包
  - 包：含有其它模块的模块
- 目录中放入名为`__init__.py`的空文件，可以采用相对于该目录的路径，来引入目录中的其它Python文件
- 包的用途
  - 把模块划分到不同的名称空间之中，使得开发者可以编写多个文件名相同的模块
  - 为外部使用者提供严谨而稳固的API，保证它们不会随着版本的变动而变动
    - 把外界可见的名称，列在名为`__all__`的特征属性里
    - `__all__`属性的值，是一份列表，其中的每个名称，都将作为本模块的一条公共API，导出给外部代码
    - 需要修改`__init__.py`文件
    - 如果只想在自己所拥有的个模块之间构建内部的API，那可能没有必要通过`__all__`来明确地导出API
- 尽量不要用`import *`



### 51. 用自编的模块定义根异常，以便将调用者与API相隔离

- 在为模块定义其API时，模块抛出的异常，与模块里定义的类和函数一样，都是接口的一部分

- 设计API时，应该自己来定义一套新的异常体系，这样会令API更加强大

- 可以在模块里面提供一种根异常，然后，令该模块所抛出的其他异常，都继承自这个根异常

  ```python
  class Error(Exception):
  	"""Base-class for all exceptions raised by this module"""
  
  class InvalidDensityError(Error):
  	"""There was a problem with a provided density value"""
  ```

- 用try/except语句来捕获异常，可以把API的调用者与模块的API相隔离

- 优点

  - 调用者可以知道调用代码是否正确
  - 帮助开发者寻找API里的bug，如果抛出了其它类型的异常，说明代码里面有bug
  - 便于API的后续演化



### 52. 用适当的方式打破循环依赖关系

- 两个模块必须相互调用对方，才能完成引入操作，就会出现循环依赖现象
- 最佳解决方案：把导致两个模块互相依赖的那部分代码，重构为单独的模块，并把它放在依赖树的底部
- 最简解决方案：执行动态的模块引入操作，既可以缩减重构所花的精力，也可以尽量降低代码的复杂度
- 引入模块的顺序
  - 当前文件目录
  - 环境变量PYTHONPATH
  - sys.path指定的路径



### 53. 用虚拟环境隔离项目，并重建其依赖关系

- 场景

  - 不同项目包依赖不同
  - 同一份代码需要在不同机器上运行

- pyvenv

  - python3.4开始，默认安装在电脑中

  - 新建名为myproject的虚拟环境

    ```
    pyvenv /tmp/myproject
    ```

  - 启用虚拟环境

    ```
    cd /tmp/myproject
    source bin/activate
    ```

  - 这时候命令行中的python，已经不再指向整个系统中的python命令，而是会指向虚拟环境目录中的那个python命令

    ```
    which python
    ```

  - 可以愉快地在虚拟环境中安装包了，不会跟系统环境冲突

  - 返回默认系统

    ```
    deactivate
    ```

  - 如果想要把这套虚拟环境复制到其它地方，可以使用

    ```
    pip freeze > requirement.txt
    ```

    把开发环境对软件包的依赖关系，明确地保存到文件之中

  - 当我们进入新的环境，可以先用pyvenv创建虚拟环境，然后使用下面的命令把相关的依赖都复制过来

    ```
    pip install -r /tmp/myproject/requirement.txt
    ```

- 如果使用python3.4之前的版本，要使用virtualenv






### 安装包

- [python有一个强大的中央仓库](https://pypi.org/)，为了安装其中的包，需要使用命令行工具pip。安装命令很简单：
  
    ```
    python2：pip install package_name
    python3：pip3 install package_name
    ```
    
    升级：
    
    ```
    pip install package_name upgrade
    ```
    
- 服务器多用户场景，非root权限

    - sudo pip install package_name --user
    - [其它方案（略麻烦）](https://zcdll.github.io/2018/01/29/own-python-pip/)

- 也可以使用setuptools：setuptools管理Python的第三方包，将包安装到site-package下，安装的包后缀一般为.egg，实际为ZIP格式。默认从 http://pypi.python.org/pypi 下载包，能够解决Python包的依赖关系；安装了setuptools之后即可用 easy_install 命令安装包，有多种安装方式可以选择

- 也可以手动安装

    - 解压安装包（pypi.org下载）到lib/site-packages下面，然后运行

        ```
        python setup.py install
        ```

        完成编译



#### 安装python

```
wget https://www.python.org/ftp/python/3.5.0/Python-3.5.0.tgz

tar -xzf Python-3.5.0.tgz

cd Python-3.5.0

mkdir -p /home/weiqing.song/software/python35

./configure --prefix="/home/weiqing.song/software/python35"

make

make install
```



#### add python 路径到PATH

```
vi .bashrc

alias mypython=/home/weiqing.song/software/python35/bin/python3

(或者 export PATH=/home/weiqing.song/software/python35/bin:$PATH）

source .bashrc

如果只用于当前终端，终端中输入：

export PATH=$PATH:/home/weiqing.song/software/python35/bin

不要修改全局变量,会影响所有用户

/etc/profile 



echo $PATH # 查看环境变量
```





#### 安装pip3

```
wget --no-check-certificate https://github.com/pypa/pip/archive/9.0.1.tar.gz

tar -zvxf 9.0.1.tar.gz

cd pip-9.0.1



mypython setup.py install

vi ~/.bashrc  # 添加下面内容

alias mypip=/home/weiqing.song/software/python35/bin/pip3

pip install --upgrade pip  # 升级 pip
```



### 远程调试

- [pycharm](https://zhuanlan.zhihu.com/p/36843200)
- vscode
- pdb, ipdb
- vim
    - https://zhuanlan.zhihu.com/p/30022074
    - https://zhuanlan.zhihu.com/p/39008816
- [pysnooper](https://zhuanlan.zhihu.com/p/63397849)



### 参考资料

- 《effective python》第七章

  