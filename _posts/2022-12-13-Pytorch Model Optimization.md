---
layout:     post
title:      Pytorch Model Optimization
subtitle:   
date:       2022-12-13
author:     bjmsong
header-img: 
catalog: true
tags:
    - 模型优化与部署 
---
## tensorboard
- https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    - 可视化数据、网络结构、loss变化、精度变化，pr曲线等
    - https://blog.csdn.net/scar2016/article/details/115361245
- https://pytorch.org/docs/stable/tensorboard.html
- https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    - 性能分析与优化
- https://medium.com/@hyoungsungkim/pytorch-tensorboard-tutorial-for-a-beginner-b037ee66574a
- https://stackoverflow.com/questions/53736966/how-to-profiling-layer-by-layer-in-pytroch

## OPTIMIZING VISION TRANSFORMER MODEL FOR DEPLOYMENT
- https://pytorch.org/tutorials/beginner/vt_tutorial.html
    - DeiT：a vision transformer model that requires a lot less data and computing resources for training to compete with the leading CNNs in performing image classification
        - Data augmentation that simulates training on a much larger dataset;
        - Native distillation that allows the transformer network to learn from a CNN’s output.
    - Quantize -> script -> Optimize -> Serialize(Lite Interpreter)
- https://pytorch.org/tutorials/recipes/script_optimized.html
    - TorchScript is actually a subset of Python and to make script work, the PyTorch model definition must only use the language features of that TorchScript subset of Python
        - https://pytorch.org/docs/master/jit_language_reference.html#language-reference
            - covers all the details of what is supported in TorchScript
    - Fix Common Errors When Using the script Method
    - Optimize a TorchScript Model
        - By default, "optimize_for_mobile" will perform the following types of optimizations:
            - Conv2D and BatchNorm fusion which folds Conv2d-BatchNorm2d into Conv2d;
            - Insert and fold prepacked ops which rewrites the model graph to replace 2D convolutions and linear ops with their prepacked counterparts.
            - ReLU and hardtanh fusion which rewrites graph by finding ReLU/hardtanh ops and fuses them together.
            - Dropout removal which removes dropout nodes from this module when training is false. 
            
## https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations

## https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
- https://zhuanlan.zhihu.com/p/83419913

## GROKKING PYTORCH INTEL CPU PERFORMANCE FROM FIRST PRINCIPLES
- https://pytorch.org/tutorials/intermediate/torchserve_with_ipex.html

## 
- 《DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion》