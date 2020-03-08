---
layout:     post
title:      Linux命令行与shell脚本编程大全
subtitle:   
date:       2020-03-08
author:     bjmsong
header-img: img/cs/linux/linux.png
catalog: true
tags:
    - Linux
---



### 一.初识Linux Shell

- Linux可以划分为四个部分
  - Linux内核
  - GNU工具
  - 图形化桌面环境
  - 应用软件



####  LInux内核

- Linus开发了第一版Linux内核
- 负责以下四种功能：
  - 系统内存管理
    - 物理内存
    - 虚拟内存
  - 软件程序管理
    - 进程：运行中的程序
  - 硬件设备管理
  - 文件系统管理



#### GNU工具

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



#### LInux桌面环境

- 完成工作的方式不止一种，Linux一直以来都以此闻名
- X Window系统
- KDE桌面
- GNOME桌面
- Unity桌面



#### Linux发行版

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



### 二. 走进shell



### 三. 基本的bash shell命令

- /etc/passwd：包含了所有系统用户账户列表以及每个用户的基本配置信息
- shell手册：在想要查找的工具的名称前面输入man命令
  - 点击空格键进行翻页，使用回车键逐行查看，点击q退出



#### 文件系统

- Linux虚拟目录结构只包含一个称为根（root）目录的基础目录：/
- 挂载点（mount point）：虚拟目录中用于分配额外存储设备的目录
- 绝对路径 
- 相对路径
  - . 表示当前目录
  - .. 表示当前目录的父目录

- ls
  - -F ： 区分文件和目录
  - -a :   隐藏文件也显示出来
  - -l ： 长列表输出
  - -l  [script] ：过滤器

- 通配符
  - ? : 代表一个字符
  - *：代表零个或多个字符



#### 处理文件

- 创建文件：touch
- 复制：cp
  - -i ： 强制询问是否要覆盖已存在文件
  - -R：递归地复制整个目录的内容
  - scp ：在不同的Linux系统之间来回copy文件 
- 制表键自动补全
- 链接文件
  - 虚拟的副本
- 重命名：mv
- 删除：rm
  - -f: 强行删除



#### 处理目录

- 





### 四. 更多的bash shell命令



### 五. 理解shell



### 六.使用Linux环境变量

- 



### 七. 理解Linux文件权限



### 八. 管理文件系统



### 九. 安装软件程序



### 十. 使用编辑器



### 十一. 构建基本脚本



### 十二. 使用结构化命令



### 十三. 更多的结构化命令



### 十四. 处理用户输入





### 参考资料



