---
layout:     post
title:      Applications of Parallel Computers
subtitle:   CS267
date:       2022-11-15
author:     bjmsong
header-img: 
catalog: true
tags:
    - 并行计算
---
- https://sites.google.com/lbl.gov/cs267-spr2021
- https://www.bilibili.com/video/BV1nA411V7iZ/?spm_id_from=333.337.search-card.all.click&vd_source=7798c62f92ce545f56fd00d4daf55e26
- 课程很新(Spring 2021)，老师讲得很清晰
- 先看ppt，准备相关知识背景，再看视频，效率会更高

## 1. Introduction & Overview
- 并行计算的目的：加速
- Parallel Computer
  + Shared Memory(SMP) or Multicore
    * connect multiple processors to a single memory system
    * contain multiple processors on a single chip
  + High Performance Computing(HPC) or Distributed Memory
  + Single Instruction Multiple Data(SIMD)
    * multiple processors(or functional units) that perform same operation on multiple data elements at once
    * most single processors have SIMD units with ~2-8 way parallelism
    * GPU use it as well
- 区别
  + 并发 vs 并行
    * Concurrency: multiple tasks are logically active at one time
    * Parallelism: multiple tasks are actually active at one time
  + Parallel Computer vs. Distributed System
    * Distributed System：机器是分布式的，不一定是为了加速计算
    * Parallel Computer：为了加速计算，可能会用到分布式的技术
- Fastest Computers in the world: top500.org
  + 技术迭代：Vector -> SIMD -> SMP -> Constellation -> MMP -> Cluster x86 -> Accelerated
- Units of Measure for HPC
  + Flop: floating point operation, usually double precision unless noted
  + Flop/s: floating point operations per second
  + GB->TB->PB
- SCIENCE USING HIGH PERFORMANCE COMPUTING
  + The Fifth Paradigm of Science: Theory,Experiment,Simulation,Data Analysis,Machine Learning
- The Motifs of Parallel Computing
  + 7 "dwarfs" of scientific computing in simulation
    * Dense Linear Algebra
    * Sparse Linear Algebra
    * Particle Methods
    * Structured Grids
    * Unstructured Grids
    * Spectral Methods(e.g. FFT)
    * Monte Carlo
- Why all computers are Parallel Computers
  + 处理器的历史发展(before 2000)
    * 摩尔定律
    * 晶体管尺寸每缩小x倍: 时钟频率提升x倍，单位面积的晶体管数量提升x^2倍，Parallel Computers提升x^4被
  + Limits
    * 数据传输时间不可能超过光速
    * Heat Density
    * 晶体管尺寸不可能小于原子
  + 

## 2. Memory Hierarchies and Matrix Multiplication
- 

## 3. More MatMul and the Roofline Performance Model
- 

## 4. Shared Memory Parallelism
- 
