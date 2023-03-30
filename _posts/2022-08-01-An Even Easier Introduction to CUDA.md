---
layout:     post
title:      An Even Easier Introduction to CUDA
subtitle:   
date:       2022-08-01
author:     bjmsong
header-img: img/ai.jpg
catalog: true
tags:
    - 并行计算
---

## Example: Adds the elements of two arrays in C++

```c++
#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add(N, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] x;
  delete [] y;

  return 0;
}
```



## Kernel function

runs on the GPU and can be called from CPU code.

```c++
// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}
```

launch the kernel

```c++
add<<<1, 1>>>(N, x, y);
```



## Memory Allocation in CUDA

```c++
// Allocate Unified Memory -- accessible from CPU or GPU
float *x, *y;
cudaMallocManaged(&x, N*sizeof(float));
cudaMallocManaged(&y, N*sizeof(float));

...

// Free memory
cudaFree(x);
cudaFree(y);
```



## CUDA Implementation

```c++
#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
```

编译

```shell
nvcc add.cu -o add_cuda
```



## Profile

```shell
$ nvprof ./add_cuda
==3355== NVPROF is profiling process 3355, command: ./add_cuda
Max error: 0
==3355== Profiling application: ./add_cuda
==3355== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  463.25ms         1  463.25ms  463.25ms  463.25ms  add(int, float*, float*)
...
```



## 多线程

GPUs run kernels using blocks of threads that are a multiple of 32 in size. (以warp为单位？)

增加线程数

```c++
add<<<1, 256>>>(N, x, y);
```

修改`kernel function`，每个线程跑一个`kernel function`

```c++
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;   // index of the current thread within its block
  int stride = blockDim.x;   // number of threads in the block
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
```

速度有明显提升（463.25ms -> 2.7ms）

```shell
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  2.7107ms         1  2.7107ms  2.7107ms  2.7107ms  add(int, float*, float*)
```



## Out of the Blocks

CUDA GPUs have many parallel processors grouped into **Streaming Multiprocessors**, or SMs. Each SM can run multiple concurrent thread blocks.  

As an example, a Tesla P100 GPU based on the [Pascal GPU Architecture](https://developer.nvidia.com/blog/inside-pascal/) has 56 SMs, each capable of supporting up to 2048 active threads. 

例如，需要用N个线程来处理N个数据，每个`block`有256个线程，`block`的数量(`numBlocks`)可以通过下面计算得到。

```c++
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;    // 避免N/blockSize除不尽，因此多一个block
add<<<numBlocks, blockSize>>>(N, x, y);
```

如下图所示，一组`block`构成一个`grid`，CUDA提供了以下参数：

- `gridDim.x`：contains the number of blocks in the grid
- `blockIdx.x`：contains the index of the current thread block in the grid
- `blockDim.x`:  number of threads in the block
- `threadIdx.x`:  index of the current thread within its block

<ul> 
<li markdown="1">
![]({{site.baseurl}}/img/cuda/1.png) 
</li> 
</ul> 

考虑到多个`block`的情况，`kernel function`改写为：

```c++
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
```

速度再次得到了提升

```shell
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  94.015us         1  94.015us  94.015us  94.015us  add(int, float*, float*)
```