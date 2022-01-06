---
layout:     post
title:      labuladong的笔记
subtitle:   
date:       2021-12-18
author:     bjmsong
header-img: img/cs/cs.jpeg
catalog: true
tags:
    - 计算机
---
## 文件系统
- /bin /sbin
    - /bin 是binary的缩写，存放着可执行文件、可执行文件的链接：cp，chmod，cat，date等常用命令
    - /sbin 是system binary的缩写，这里存放的命令可以对系统配置进行操作。普通用户可以使用这些命令查看系统状态(如ipconfig)，超级用户（sudo）可以使用这些命令更改配置
- /boot
    - 系统启动需要的文件
- /dev
    - 存放所有的设备文件（硬盘、鼠标、键盘。。）。在Linux中，所有东西都是以文件的形式存在的，包括硬件设备
- /etc
    - 存放很多程序的配置信息
- /lib
    - Library的简写，包含bin和sbin中可执行文件的依赖
- /media
    - 这里会有一个以用户名命名的文件夹，里面是自动挂载的设备，比如U盘、移动硬盘、网络设备等
- /mnt
    - 手动挂载设备的地方
- /proc
    - 正在运行程序的状态信息
- /usr
    - 非系统必须的资源，比如用户安装的应用程序
- /opt
    - 使用比较随意，一般来说我们自己在浏览器上下载的软件，安装在这里比较好。当然，包管理工具下载的软件也可能被存放在这里
- /root
    - 用户的家目录，普通用户需要授权才能访问
- /var
    - 这个名字是历史遗留的，现在该目录最主要的作用是存储日志（log）信息，比如说程序崩溃，防火墙检测到异常等等信息都会记录在这里
    - 日志文件不会自动删除，也就是说随着系统使用时间的增长，你的var目录占用的磁盘空间会越来越大，也许需要适时清理一下
- 如果修改系统配置，就去 /etc 找，如果修改用户的应用程序配置，就在用户家目录的隐藏文件里找
- 你在命令行里可以直接输入使用的命令，其可执行文件一般就在以下几个位置：
/bin    
/sbin
/usr/bin
/usr/sbin
/usr/local/bin
/usr/local/sbin
/home/USER/.local/bin
/home/USER/.local/sbin
- 如果你写了一个脚本/程序，想在任何时候都能直接调用，可以把这个脚本/程序添加到上述目录中
- 如果某个程序崩溃了，可以到 /var/log 中尝试寻找出错信息，到 /tmp 中寻找残留的临时文件
- 设备文件在 /dev 目录，但是一般来说系统会自动帮你挂载诸如 U 盘之类的设备，可以到 /media文件夹访问设备内容



## 进程、线程和文件描述符
- 在 Linux 系统中，进程和线程几乎没有区别
- 
- 
- 

## Linux shell
- 

## session和cookie
- 

## 加密算法的前世今生
- 

## Git
- 

## 参考资料
https://labuladong.gitee.io/algo/5/34/