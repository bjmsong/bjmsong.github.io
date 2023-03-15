---
layout:     post
title:      深度学习推理引擎`KuiperInfer`源码分析
subtitle:   之一
date:       2023-03-03
author:     bjmsong
header-img: img/kuiper/logo.jpg
catalog: true
tags:
    - 深度学习推理系统
---


## 概览

`KuiperInfer`(**[项目地址](https://github.com/zjhellofss/KuiperInfer)**)是一个开源的深度学习推理引擎，感谢作者提供了一个这么好的项目，供大家学习。本文分享一下我对这个项目的理解，欢迎交流。

训练好的深度学习模型，需要通过推理框架部署到不同的设备上，高效完成模型推理，服务应用场景。与训练框架不同的是，深度学习推理框架**没有梯度反向传播功能**，因为算法模型文件中的权重系数已经被固化，推理框架只需要读取、加载并完成对新数据的预测即可。

作为深度学习推理框架的核心组件，**推理引擎的整体流程**如下图所示：

![1676960146688](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1676960146688.png)

下面介绍下`KuiperInfer`的核心模块。



## 张量

**张量是存储数据(例如输入、输出、系数或参数)的主要容器**。张量是一种递归和自包含的定义，比如：4维Tensor由N个3维Tensor组成，3维Tensor由N个2维Tensor组成，2维Tensor由N个1维的Tensor组成，1维Tensor由N个0维Tensor组成，0维Tensor维为标量。 

典型的图像数据RGB为3维Tensor，RGB数据的保存方式有RGBRGBRGB....或者 RRR...GGG...BBB...，即NHWC或者NCHW，如下图所示。`KuiperInfer`采用的是**NCHW**格式（NCHW分别表示批次、通道和高宽）。

![img](https://gitee.com/cao_fx/KuiperInfer_Better/raw/master/02/NCHW_NHWC.png)





### `Tensor`类的设计

成员变量：

```c++
 private:
  // 行优先的shape
  void ReView(const std::vector<uint32_t>& shapes);
  std::vector<uint32_t> raw_shapes_;  // 张量数据的实际尺寸大小
  arma::fcube data_;                  // 张量数据
```

`KuiperInfer`的张量以**Armadillo**类中的**`cube`**(三维矩阵)作为数据的container，在`cube`之上实现了`Tensor`的接口，一个`cube`由多个**`mat`**（二维矩阵）在内存中连续存储组成。

- **Armadillo**是一个接口友好，高性能的线性代数库，底层可以调用`OpenBlas`、`MKL`。

![1676976449108](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1676976449108.png)

张量是**逻辑上的多维数组，底层数据结构为一维数组（内存连续）**

张量类提供的**数据读取**方法有：

| rows      | cols        | channels         | size     | empty  |
| --------- | ----------- | ---------------- | -------- | ------ |
| **index** | **shapes**  | **raw_shapes**   | **data** | **at** |
| **Show**  | **raw_ptr** | **TensorIsSame** |          |        |

张量类提供的**数据操作**方法有：

| set_data            | Padding              | Fill                      | Ones             | Rand      |
| ------------------- | -------------------- | ------------------------- | ---------------- | --------- |
| **ReRawshape**      | **ReRawView**        | **Flatten**               | **Transform**    | **Clone** |
| **TensorBroadcast** | **TensorElementAdd** | **TensorElementMultiply** | **TensorCreate** |           |

张量可以通过加载csv文件生成

```c++
class CSVDataLoader {
 public:
    // 从csv文件中初始化张量
  static arma::fmat LoadData(const std::string &file_path, char split_char = ',');

 private:
    // 得到csv文件的尺寸大小，LoadData中根据这里返回的尺寸大小初始化返回的fmat
  static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);
};
```



### 列主序

`mat`类是列主序的，也就是同一列数据存放在内存中相邻的位置。这个特性会影响很多对`Tensor`的操作。

![1676966822021](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1676966822021.png)

例如`Fill(vector<float>values)`方法：以`values`中的数据去填充`Tensor`。如果将顺序的一组数据`[0,1,2,3,4,5,...,15]`填充到一个大小为4×4的`Tensor`中。默认情况下填充的结果是这样的：

![1677489471946](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1677489471946.png)

如果想要实现行主序的填充效果，需要对填充结果进行**转置**。

![1677489575411](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1677489575411.png)

还有`Reshape`方法：调整`tensor`的形状。默认的reshape结果是这样的：

![1677489744205](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1677489744205.png)

![1677489755430](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1677489755430.png)

如果想要实现行主序的reshape，不能直接调用`cube`的方法，只能**通过位置计算的方式来对逐个元素进行搬运**。

![1677489935307](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1677489935307.png)

```c++
void Tensor<float>::ReView(const std::vector<uint32_t>& shapes) {
  CHECK(!this->data_.empty());
  const uint32_t target_channels = shapes.at(0);
  const uint32_t target_rows = shapes.at(1);
  const uint32_t target_cols = shapes.at(2);
  arma::fcube new_data(target_rows, target_cols, target_channels);

  const uint32_t plane_size = target_rows * target_cols;
  for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
    const arma::fmat& channel = this->data_.slice(c);
    for (uint32_t c_ = 0; c_ < this->data_.n_cols; ++c_) {
      const float* colptr = channel.colptr(c_);
      for (uint32_t r = 0; r < this->data_.n_rows; ++r) {
        const uint32_t pos_index =
            c * data_.n_rows * data_.n_cols + r * data_.n_cols + c_;
        const uint32_t ch = pos_index / plane_size;
        const uint32_t row = (pos_index - ch * plane_size) / target_cols;
        const uint32_t col = (pos_index - ch * plane_size - row * target_cols);
        new_data.at(row, col, ch) = *(colptr + r);
      }
    }
  }
  this->data_ = new_data;
}
```





## 计算图

计算图是神经网络的中间表达。计算图根据训练好的神经网络结构，将Tensor和计算节点（Operator）有效的组织和连接形成一个整体，形成一个有向无环图（DAG），并描述如何将输入的数据通过各种层进行运算得到输出。

![1676959793348](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1676959793348.png)

![img](https://gitee.com/cao_fx/KuiperInfer_Better/raw/master/02/graph_structure.png)



### 计算图(`RuntimeGraph`)

`RuntimeGraph`类负责计算图的初始化、构建、执行等，主要方法如下：

```c++
// 计算图的初始化
bool Init();

// 构建计算图
void Build(const std::string& input_name, const std::string& output_name);  

// 计算图的执行
std::vector<std::shared_ptr<Tensor<float>>> Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs, bool debug = false);

// 根据计算图中的计算节点来生成Layer
static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator>& op);

// 检查当前节点是否就绪
static bool CheckOperatorReady(const std::shared_ptr<RuntimeOperator>& op);

// 探查下一层的计算节点
static void ProbeNextLayer(
      const std::shared_ptr<RuntimeOperator>& current_op,
      std::deque<std::shared_ptr<RuntimeOperator>>& operator_queue,
      const std::vector<std::shared_ptr<Tensor<float>>>& layer_output_data);
```

成员变量如下：

```c++
  enum class GraphState {     
    NeedInit = -2,
    NeedBuild = -1,
    Complete = 0,
  };
  GraphState graph_state_ = GraphState::NeedInit;  // 计算图构建状态
  std::string input_name_;   /// 计算图输入节点的名称
  std::string output_name_;  /// 计算图输出节点的名称
  std::string param_path_;   /// 计算图的结构文件
  std::string bin_path_;     /// 计算图的权重文件
  std::map<std::string, std::shared_ptr<RuntimeOperator>>
      input_operators_maps_;  /// 保存输入节点
  std::map<std::string, std::shared_ptr<RuntimeOperator>>
      output_operators_maps_;  /// 保存输出节点
  std::vector<std::shared_ptr<RuntimeOperator>>
      operators_;                       /// 计算图的计算节点
  std::unique_ptr<pnnx::Graph> graph_;  /// pnnx的graph
```



### 计算节点（`RuntimeOperator`）

计算节点是计算图中的一个节点，用来执行特定计算。如`Relu`，卷积、池化等。

计算节点主要包含三部分内容：

1. 节点的输入、输出数据，节点的参数（如卷积核的大小），节点的权重（如卷积核的weight、bias）等；
2. 后继计算节点和前驱计算节点；
3. 层(Layer): 计算过程的具体执行者

`RuntimeOperator`类设计如下：

```c++
struct RuntimeOperator {
  int32_t meet_num = 0;  /// 计算节点被相连接节点访问到的次数
  virtual ~RuntimeOperator() {
    for (auto& param : this->params) {
      if (param.second != nullptr) {
        delete param.second;
        param.second = nullptr;
      }
    }
  }
  std::string name;              /// 计算节点的名称
  std::string type;              /// 计算节点的类型
  std::shared_ptr<Layer> layer;  /// 节点对应的计算Layer

  std::vector<std::string> output_names;  /// 节点的输出节点名称
  std::shared_ptr<RuntimeOperand> output_operands;  /// 节点的输出操作数

  std::map<std::string, std::shared_ptr<RuntimeOperand>>
      input_operands;  /// 节点的输入操作数
  std::vector<std::shared_ptr<RuntimeOperand>>
      input_operands_seq;  /// 节点的输入操作数，顺序排列
  std::map<std::string, std::shared_ptr<RuntimeOperator>>
      output_operators;  /// 输出节点的名字和节点对应

  std::map<std::string, RuntimeParameter*> params;  /// 算子的参数信息
  std::map<std::string, std::shared_ptr<RuntimeAttribute>>
      attribute;  /// 算子的属性信息，内含权重信息
};
```



### 操作数（`RuntimeOperand`）

操作数是每个节点的输入和输出数据，`RuntimeOperand`类设计如下：

```c++
struct RuntimeOperand {
  std::string name;                                     /// 操作数的名称
  std::vector<int32_t> shapes;                          /// 操作数的形状
  std::vector<std::shared_ptr<Tensor<float>>> datas;    /// 存储操作数
  RuntimeDataType type = RuntimeDataType::kTypeUnknown; /// 操作数的类型，一般是float
};
```



### PNNX

[PNNX](https://link.zhihu.com/?target=https%3A//github.com/Tencent/ncnn/tree/master/tools/pnnx) （PyTorch Neural Network eXchange）是PyTorch模型互操作性的开放标准。PNNX为PyTorch提供了一种开源的模型格式，它定义了与Pytorch相匹配的数据流图和运算图。Pytorch训练好一个模型之后，模型可以转换到pnnx格式文件，通过读取pnnx格式文件，形成计算图。

ONNX作为广泛应用的模型中间表达，具有以下一些问题：

- ONNX以ProtoBuffer作为模型表达的文件格式，对数据传输友好，但是可读性不友好，很难直接修改计算图
- 算子的定义和PyTorch不完全兼容，需要用很多小算子去拼接，使得计算图变得过于复杂，同时降低推理效率
- 因为ONNX要适配不同的深度学习框架，添加了大量的参数，增加了开发者负担

PNNX具有以下特性：

- 模型文件用户可读，容易修改

  ```PNNX
  7767517
  4 3
  pnnx.Input      input       0 1 0
  nn.Conv2d       conv_0      1 1 0 1 bias=1 dilation=(1,1) groups=1 in_channels=12 kernel_size=(3,3) out_channels=16 padding=(0,0) stride=(1,1) @bias=(16)f32 @weight=(16,12,3,3)f32
  nn.Conv2d       conv_1      1 1 1 2 bias=1 dilation=(1,1) groups=1 in_channels=16 kernel_size=(2,2) out_channels=20 padding=(2,2) stride=(2,2) @bias=(20)f32 @weight=(20,16,2,2)f32
  pnnx.Output     output      1 0 2
  ```

- 算子跟PyTorch Python API完全对应

  ![1677551646971](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1677551646971.png)

- expression operator

  - 完整的算术表达式，阅读方便，减少访存

- 计算图优化

- 支持Pytorch自定义算子

- Tensor shape propagation

- ....



### 计算图的构建步骤

#### 加载PNNX模型文件，生成PNNX计算图

```c++
RuntimeGraph graph(param_path, weight_path);
```



#### 通过PNNX的operators，构建计算节点

```c++
InitGraphOperatorsInput();
InitGraphOperatorsOutput();
InitGraphAttrs();    
InitGraphParams();
```



#### 创建计算节点对应的Layer层

```c++
std::shared_ptr<Layer> RuntimeGraph::CreateLayer(
    const std::shared_ptr<RuntimeOperator>& op) {
  LOG_IF(FATAL, !op) << "Operator is empty!";
  const auto& layer = LayerRegisterer::CreateLayer(op);
  LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
  return layer;
}
```



原本不同的layer层需要通过调用对应的layer类来构造，例如：

```c++
ReluLayer layer(relu_op);
SigmoidLayer layer(sigmoid_op);
.....
```



这里通过统一的**工厂方法**，实现了不同layer层的生成。步骤如下：

- 每个Layer会通过定义`LayerRegistererWrapper`对象来调用`RegisterCreator`方法：**注册器模式**

```c++
LayerRegistererWrapper kSigmoidGetInstance("nn.Sigmoid", SigmoidLayer::GetInstance);

class LayerRegistererWrapper {
 public:
  LayerRegistererWrapper(const std::string &layer_type, const LayerRegisterer::Creator &creator) {
    LayerRegisterer::RegisterCreator(layer_type, creator);
  }
```

- `RegisterCreator`方法通过维护一个**静态的注册表**来注册layer。注册表是一个Map，key是对应的OpType，用来查找对应的value，value是用于创建该层的对应方法(Creator)：**单例模式**。

```c++
typedef std::map<OpType, Creator> CreateRegistry;

void LayerRegisterer::RegisterCreator(const std::string &layer_type,
                                      const Creator &creator) {
  CHECK(creator != nullptr);
  CreateRegistry &registry = Registry();
  CHECK_EQ(registry.count(layer_type), 0)
      << "Layer type: " << layer_type << " has already registered!";
  registry.insert({layer_type, creator});
}

LayerRegisterer::CreateRegistry &LayerRegisterer::Registry() {
  static CreateRegistry *kRegistry = new CreateRegistry();
  CHECK(kRegistry != nullptr) << "Global layer register init failed!";
  return *kRegistry;
}
```

- 最后，只需要传入对应的`RuntimeOperator`，调用`CreateLayer`方法就可以生成对应的Layer，十分优雅。

```c++
std::shared_ptr<Layer> LayerRegisterer::CreateLayer(
    const std::shared_ptr<RuntimeOperator> &op) {
  CreateRegistry &registry = Registry();
  const std::string &layer_type = op->type;
  LOG_IF(FATAL, registry.count(layer_type) <= 0)
      << "Can not find the layer type: " << layer_type;
  const auto &creator = registry.find(layer_type)->second;

  LOG_IF(FATAL, !creator) << "Layer creator is empty!";
  std::shared_ptr<Layer> layer;
  const auto &status = creator(op, layer);
  LOG_IF(FATAL, status != ParseParameterAttrStatus::kParameterAttrParseSuccess)
      << "Create the layer: " << layer_type
      << " failed, error code: " << int(status);
  return layer;
}
```



#### 关联计算节点的后继计算节点

```c++
for (const auto& current_op : this->operators_) {
    // 获取当前节点的所有后继节点names
    const std::vector<std::string>& output_names = current_op->output_names;
    for (const auto& next_op : this->operators_) {
        if (next_op == current_op) {
            continue;
        }
        // 如果其余节点的name符合当前节点的后继节点names，则将这个其余节点作为当前节点的后继
        if (std::find(output_names.begin(), output_names.end(), next_op->name) !=
            output_names.end()) {
            current_op->output_operators.insert({next_op->name, next_op});
        }
    }
}
```



#### 预分配操作数内存

根据计算图，可以得到算子的输入输出操作数大小，可以在计算图执行之前提前分配好内存

由于每个节点的输入就是上一层节点的输出，因此除了输入节点之外，其他节点的输入可以复用它上一层节点的输出空间



### 计算图的调度执行

Graph在执行时在逻辑上可以分为两条路径，一条是**控制流**，另外一条是**数据流**。在数据流中，前一个`operator`产生的输出传递到后继`operator`作为输入。

![1677568136017](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1677568136017.png)

每个计算节点必须要等它依赖的节点完成计算，才能进行计算，是一个**拓扑排序**的过程。也就是进行**广度优先遍历**，通过一个队列维护要遍历的计算节点，当某个计算节点的前驱节点都已加入队列中，则将该节点也加入到队列中。

![1676778404291](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1676778404291.png)

寻找并拷贝上一级的输出到后继节点

```c++
void RuntimeGraph::ProbeNextLayer(
    const std::shared_ptr<RuntimeOperator> &current_op,
    std::deque<std::shared_ptr<RuntimeOperator>> &operator_queue,
    std::vector<std::shared_ptr<Tensor<float>>> layer_output_datas) {
  const auto &next_ops = current_op->output_operators;

  std::vector<std::vector<std::shared_ptr<ftensor>>> next_input_datas_arr;
  for (const auto &next_op : next_ops) {
    const auto &next_rt_operator = next_op.second;
    const auto &next_input_operands = next_rt_operator->input_operands;
    // 找到后继节点
    if (next_input_operands.find(current_op->name) !=
        next_input_operands.end()) {
      std::vector<std::shared_ptr<ftensor>> next_input_datas =
          next_input_operands.at(current_op->name)->datas;
      next_input_datas_arr.push_back(next_input_datas);
      next_rt_operator->meet_num += 1;
      if (std::find(operator_queue.begin(), operator_queue.end(),
                    next_rt_operator) == operator_queue.end()) {
        if (CheckOperatorReady(next_rt_operator)) {
          operator_queue.push_back(next_rt_operator);
        }
      }
    }
  }
  SetOpInputData(layer_output_datas, next_input_datas_arr);
}
```





## 参考资料

- https://github.com/zjhellofss/KuiperInfer

- https://zhuanlan.zhihu.com/p/593215728

- https://blog.csdn.net/qq_32901731/category_12176352.html

- [PNNX：PyTorch Neural Network Exchange](https://www.bilibili.com/video/BV1Uv411u78D/?spm_id_from=333.999.0.0&vd_source=7798c62f92ce545f56fd00d4daf55e26)

- https://gitee.com/cao_fx/KuiperInfer_Better

  

