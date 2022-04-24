---
layout:     post
title:      effective python
subtitle:   
date:       2022-04-24
author:     bjmsong
header-img: 
catalog: true
tags:
    - 
---
2. 遵循PEP8风格指南
3. 了解bytes,str与unicode的区别
- 在python3中，bytes是一种包含8位值的序列，str是一种包含Unicode字符的序列
- 把编码和解码操作放在界面最外围来做，程序的核心部分应该使用Unicode字符类型(python3中的str)
4. 用辅助函数来取代复杂的表达式

14. 尽量用异常来表示特殊情况，使得调用者可以正确处理
16. 考虑用生成器来改写直接返回列表的函数
18. 用数量可变的位置参数减少视觉干扰
19. 用关键字参数来表达可选的行为