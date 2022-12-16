---
layout:     post
title:      Linux命令行与shell脚本编程大全
subtitle:   
date:       2020-03-08
author:     bjmsong
header-img: img/cs/linux/linux.png
catalog: true
tags:
    - Computer Science
---
## 一.初识Linux Shell
- Linux可以划分为四个部分
  - Linux内核
  - GNU工具
  - 图形化桌面环境
  - 应用软件


###  Linux内核
- Linus开发了第一版Linux内核
- 负责以下四种功能
  - 系统内存管理
    - 物理内存
    - 虚拟内存
  - 软件程序管理
    - 进程：运行中的程序
  - 硬件设备管理
  - 文件系统管理
- https://www.bilibili.com/video/BV1cV41117jP/?spm_id_from=333.999.0.0&vd_source=7798c62f92ce545f56fd00d4daf55e26
- 提供了API，用户可以进行系统调用
- 内核实现策略
  - 宏内核
  - 微内核
- 内核的核心模块
  - 进程的调度、切换
  - 内存管理
  - 虚拟内存机制
  - 和网络交互的地方
  - 设备驱动程序
  - 进程通信机制&锁

### GNU工具
- GNU（GNU‘s Not Unix）组织开发了一套完整的Unix工具
- 开源软件理念：允许程序员开发软件，并将其免费发布
- Linux = Linux内核 + GNU工具
- 核心GNU工具
  - 处理文件的工具
  - 操作文本的工具
  - 管理进程的工具
- shell
  - 交互式工具，为用户提供了启动程序、管理文件系统中的文件以及运行在Linux系统上的进程的途径
  - 默认是bash shell


### Linux桌面环境
- 完成工作的方式不止一种，Linux一直以来都以此闻名
- X Window系统
- KDE桌面
- GNOME桌面
- Unity桌面


### Linux发行版
- 完整的Linux系统包
- 核心Linux发行版
  - Red Hat
  - 安装对新手来说是一场噩梦
- 特定用途的Linux发行版
  - 仅包含主流发行版中一小部分用于某种特定用途的应用程序
  - 安装简单
  - Ubuntu，CentOS
- Linux LiveCD
  - 无需将Linux安装到硬盘就能体验Linux的发行版



## 二. 走进shell



## 三. 基本的bash shell命令
- shell手册：在想要查找的工具的名称前面输入`man`命令
  - 点击空格键进行翻页，使用回车键逐行查看，点击`q`退出


### 文件系统
- Linux虚拟目录结构只包含一个称为根（root）目录的基础目录：`/`
- 挂载点（mount point）：虚拟目录中用于分配额外存储设备的目录
- 绝对路径 
- 相对路径
  - `.` 表示当前目录
  - `..` 表示当前目录的父目录

- ls
  - `-F` ： 区分文件和目录
  - `-a` :   隐藏文件也显示出来
  - `-l` ： 长列表输出
  - `-l  [script]` ：过滤器

- 通配符
  - `?` : 代表一个字符
  - `*`：代表零个或多个字符



### 处理文件
- 创建文件：touch
- 复制：cp
  - `-i` ： 强制询问是否要覆盖已存在文件
  - `-R`：递归地复制整个目录的内容
  - `scp` ：在不同的Linux系统之间来回copy文件 
- 制表键自动补全
- 链接文件
  - 虚拟的副本
- 重命名：mv
- 删除：rm
  - -f: 强行删除



### 处理目录
- 创建目录：mkdir
  - -p：同时创建多个目录和子目录
- 删除目录：rm -r



### 查看文件内容
- 查看文件类型：file
- cat：查看整个文件
  - -n：加上行号
- more：显示文本文件的内容，在显示每页数据之后停下来
  - q：退出
- less：more 命令的升级版
- head
- tail



## 四. 更多的bash shell命令
### 监测程序
- ps：显示运行在当前控制台下的属于当前用户的进程
  - 参数很多
- 实时监测：top
- 结束进程
  - kill pid
  - killall ：通过进程名而不是PID来结束进程



### 监测磁盘空间
- 挂载存储媒体
- df -h：查看所有已挂载磁盘的使用情况
- du：显示某个特定目录的磁盘使用情况
  - -c：显示所有已列出文件总的大小
  - -h：按用户易读的格式输出大小



### 处理数据文件
- sort：排序
  - `du -sh * | sort -nr`
- grep：搜索
  - 在输入或者指定的文件中查找包含匹配指定模式的字符的行
- 压缩
  - gzip
    - gzip：压缩文件
    - gunzup：解压文件
  - tar：最广泛使用的归档工具
    - tar -zxvf filename.tgz
    - tar-xvf filename.tar



## 五. 理解shell
- 输入/bin/bash或其它等效的bash命令时，会创建一个新的shell程序，被称为子shell

  - 只有部分父进程的环境被复制到子shell环境中

- 外部命令

  - 存在于bash shell之外的程序，可以用which 和 type命令找到它，如

    ```
    which ps
    ```

  - 当外部命令执行时，会创建出一个子进程

- 内部命令

  - 不需要使用子进程来执行，更高效，不会受到环境变化的影响
  - history：最近使用过的命令列表
  - alias：命令别名



## 六.使用Linux环境变量
- 环境变量来存储有关shell会话和工作环境的信息
- 全局环境变量
  - 对于shell会话和所有生成的子shell都是可见的
  - 查看：env
  - 查看个别
    - printenv
    - echo $
- 局部环境变量
  - 只对创建它们的shell可见
  
## 七. 理解Linux文件权限
- Linux安全系统的核心是用户账户
- /etc/passwd：包含了所有系统用户账户列表以及每个用户的基本配置信息
- root账户是管理员
- 用标准的Linux用户管理工具去执行用户管理功能，而不是直接修改/etc/passwd文件



### 使用linux组
- 组权限允许多个用户对系统中的对象共享一组共用的权限
- /etc/group文件包含系统上用到的每个组的信息



### 理解文件权限

<ul> 
<li markdown="1"> 
![]({{site.baseurl}}/img/cs/linux/文件权限.png) 
</li> 
</ul> 

- 改变权限：chmod 
  - 以八进制值来描述权限
- 改变所属关系：chown，chgrp
- 共享文件：通过创建组



## 八. 管理文件系统



## 九. 安装软件程序



## 十. 使用编辑器



## 十一. 构建基本脚本



## 十二. 使用结构化命令



## 十三. 更多的结构化命令



## 十四. 处理用户输入



## 十五.呈现数据



## 十六. 控制脚本



## 十七. 创建函数





## 十八. 图形化桌面环境中的脚本编程



## 十九. 初识sed和gawk



## 二十. 正则表达式

- https://deerchao.cn/tutorials/regex/regex.html 
- https://www.hahack.com/wiki/sundries-regexp.html


## 二十一. sed进阶



## 二十二. gawk进阶









### 参考资料



