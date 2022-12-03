---
layout:     post
title:      【CMU】Deep Learning Systems
subtitle:   Algorithms and Implementation
date:       2022-12-01
author:     bjmsong
header-img: 
catalog: true
tags:
    - ML System
---
- https://dlsyscourse.org/
- Fall 2022
- CMU 10714
- 第一次online授课
- 已注册：gmail邮箱
- 先看ppt，再看视频（选择性看），再写code

## Introduction
- Throughout the course, students will design and build from scratch a complete deep learning library, capable of efficient GPU-based operations, automatic differentiation of all implemented functions, and the necessary modules to support parameterized layers, loss functions, data loaders, and optimizers.

## Assignments and project
- https://dlsyscourse.org/assignments/
- an introductory homwork： review / test of your background
- four major homework assignments 
    - build a basic deep learning library, comparable to a very minimal version of PyTorch or TensorFlow, scalable to a reasonably-sized system (e.g., with fast GPU implementations of operations)
- a final project(Group)
    - an implementation of a substantial new feature within the developed library, plus an implementation of a model using this feature (than runs under the developed library not, e.g., done within PyTorch/Tensorflow)
- There is no formal “credit” for taking the course, but everyone who an average of at least 80% on the homeworks, and submits a final project, will receive a certificate of completion

## 互相交流
- https://forum.dlsyscourse.org/login
     - 积极参与
- https://zhuanlan.zhihu.com/p/563035837
- https://github.com/fbsh/cmu-10714
- https://github.com/HeCheng0625/10714
- qq群

## 1 - Introduction / Logistics
- Aim of this course
    - an introduction to the functioning of modern deep learning systems
    - the underlying concepts of modern deep learning systems like automatic differentiation, neural network architectures, optimization, and efficient operations on systems like GPUs
    -  build(from scratch) needle, a deep learning library loosely similar to PyTorch, and implement many common architectures in the library
- Why study deep learning systems?
    - To build deep learning systems
    - To use existing systems more effectively
    - Deep learning systems are fun!
        - you could probably write a “reasonable” deep learning library in <2000 lines of (dense) code
- Elements of deep learning systems
    - Compose multiple tensor operations to build modern machine learning models
    - Transform a sequence of operations (automatic differentiation)
    - Accelerate computation via specialized hardware
    - Extend more hardware backends, more operators
- Being released for the first time. There will almost certainly be some bugs in the content or assignments
- Prerequisites
    - Python：熟练
    - C++: 基础特性
    - 数学
        - 线性代数：矩阵乘法
        - 微积分：微分
    - 机器学习

## 2 - ML Refresher / Softmax Regression
- 

## 3 - Manual Neural Networks / Backprop

## 4 - Automatic Differentiation

## 5 - Automatic Differentiation Implementation

## 6 - Optimization

## 7 - Neural Network Library Abstractions

## 8 - NN Library Implementation

## 9 - Normalization, Dropout, + Implementation

## 10 - Convolutional Networks

## 11 - Hardware Acceleration for Linear Algebra

## 12 - Hardware Acceleration + GPUs

## 13 - Hardware Acceleration Implementation

## 14 - Convoluations Network Implementation

## 15 - Training Large Models

## 16 - Generative Adversarial Networks


## 17 - Generative Adversarial Networks Implementation

## 18 - Sequence Modeling + RNNs

## 19 - Sequence Modeling Implementation

## 20 - Transformers + Attention

## 21 - Transformers + Attention Implementation

## 22 - Implicit Layers

## 24 - Model Deployment

## 25 - Machine Learning Compilation and Deployment Implementation

## 26 - Future Directions / Q&A 