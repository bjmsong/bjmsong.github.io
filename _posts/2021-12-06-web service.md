---
layout:     post
title:      web service
subtitle:   
date:       2021-12-06
author:     bjmsong
header-img: 
catalog: true
tags:
    - Computer Science
---
- 顾名思义就是基于Web的服务。它使用Web(HTTP)方式，接收和响应外部系统的某种请求，从而实现远程调用。可以将你的服务(一段代码)发布到互联网上让别人去调用,也可以调用别人机器上发布的Web Service,就像使用自己的代码一样
- HTTP（超文本传输协议）
  - 保证客户机和服务器之间的通信
  - http method
    - GET:从指定的资源请求数据
    - POST:向指定的资源提交要被处理的数据
    - PUT、HEAD。。。。
  - https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol
- 使用方式
  - Rest
    - Representational State Transfer
    - It is a software architectural style
    - RESTful: Any software that follows the REST architectural style
    - allows for more effective communication between clients (code that handles user interaction and user interface) and servers (code that sends data to the client)
    - 四种操作：PUT (update data), GET (retrieve data), POST (create new data) and DELETE (delete data)
  - RPC 
    - Remote-Procedure-Call , 远程过程调用
    - 计算机通信协议
    - 允许一台计算机的程序调用另一台计算机的子程序
    - 允许的数据带宽有限
  - SOAP，Simple Object Access Protocol(简单对象访问协议)
  - 资料
    - https://www.jianshu.com/p/7d6853140e13
    - https://medium.com/apis-you-wont-hate/understanding-rpc-rest-and-graphql-2f959aadebe7
    - https://medium.com/@donovanso/what-is-a-restful-web-service-a5280f795c89
    - https://medium.com/@parastripathi/what-is-rest-api-a-beginners-guide-700e4931e67c
    - https://blog.jscrambler.com/rpc-style-vs-rest-web-apis/
    - https://zhuanlan.zhihu.com/p/34440779
    - https://www.hardikp.com/2018/07/28/services/
    - https://www.cnblogs.com/softidea/p/7232035.html
- 数据格式
  - JSON
  - XML， Extensible Markup Language（扩展性标记语言）
- API
  - an intermediate between the client and server
  - a server provides many functionalities, but as a client you do not know how many services a server provides or what are the functionalities that you can utilize? To solve this problem you need API.
  - **All Web services are APIs but not all APIs are web services**
- web server
  - 一般来说，server 有两重意思
    - 有时候 server 表示硬件，也就是一台机器。它还有另一个名字：「主机」。
      **更多时候，server 表示软件程序，这种程序主要用来对外提供某些服务，比如邮件服务、FTP 服务、数据库服务、网页服务等。**
    - 作为开发者，我们说 server 的时候，一般指的后者，也就是一个 24 小时运行的软件程序。
  - NGINX、Apache、Node.js、tomcat 、Gunicorn... 
- web 应用框架
  - Django：使得开发复杂的、数据库驱动的网站变得简单
    https://www.cnblogs.com/feixuelove1009/p/5823135.html
    https://www.zhihu.com/question/65502802
  - Flask
      - http://docs.jinkan.org/docs/flask/
      - https://medium.com/bhavaniravi/build-your-1st-python-web-app-with-flask-b039d11f101c
      - https://www.youtube.com/watch?v=MwZwr5Tvyxo&list=PL-osiE80TeTs4UjLw5MM6OjgkjFeUxCYH&ab_channel=CoreySchafer
      - https://github.com/CoreyMSchafer/code_snippets/tree/master/Python/Flask_Blog
  - FastAPI
