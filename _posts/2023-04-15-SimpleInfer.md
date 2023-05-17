---
layout:     post
title:      SimpleInfer
subtitle:   
date:       2023-04-15
author:     bjmsong
header-img: img/ai.jpg
catalog: true
tags:
    - 模型推理
---

## 总览

如项目名字所言，对`Kupinfer`进行了大量的简化。代码简洁，同时把需要安装的第三方库做了替换，可以直接加载头文件，或者从源码编译，避免了本地安装，减少了环境依赖。

```
glog -> abseil
google test -> Catch2
armadillo -> Eigen
OpenCV -> simpleocv
```



## Tensor

本项目用到了`Eigen`库的[`Tensor`](https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html)类，默认是行主序的，替代`armadillo`的`cube`类。`Eigen`提供的方法丰富，基本可以满足对`Tensor`的操作，因此不需要给`Tensor`创建新的方法。

```c++
// 模板：T表示数据类型，num_indices表示张量是几维的， Options默认是1，表示行主序
// Eigen::TensorMap 允许你将一个指向连续内存的指针（或者可以通过指针访问的对象）与 Eigen::Tensor 类型相关联。
// 通过这样的映射，你可以在 Eigen 的张量操作中使用这个内存块，而不需要额外的拷贝操作。
template<typename T, int num_indices, int Options = 0x1>
using EigenTensorMap = Eigen::TensorMap<EigenTensor<T, num_indices, Options>>;

template<typename T, int num_indices, int Options = 0x1>
EigenTensorMap<T, num_indices> GetEigenTensor() const {
    assert(IsSameDataType<T>(data_type_));

    return EigenTensorMap<T, num_indices, Options>(static_cast<T*>(data_),
    		ToEigenDSize<num_indices>(shape_));
}
```

其中，`data_`直接从堆上分配空间，因此需要手动管理内存。

```c++
// 给data_分配空间
Status Tensor::Allocate() {
    int total_size = ElementSize(data_type_);
    for (const auto s : shape_) {
        total_size *= s;
    }

    if (total_size > 0) {
        data_ = malloc(total_size);
        if (nullptr != data_) {
            use_internal_data_ = true;
            
            return Status::kSuccess;
        }
    }

    return Status::kFail;
}

// 释放data_的空间
Status Tensor::Deallocate() {
    if (use_internal_data_) {
        if (nullptr != data_) {
            free(data_);
            data_ = nullptr;
        }

        use_internal_data_ = false;
        
        return Status::kSuccess;
    }

    return Status::kFail;
}
```



## Layer

### 统一创建

`Layer`的统一创建也是使用工厂模式，首先是注册`Layer`:

```c++
// 注册表，维护所有的layer
// 静态全局变量在程序加载到内存中时，就完成初始化了
static std::map<std::string, LayerRegistryEntry> layer_registry_map = {
    LAYER_REGISTRY_ITEM(nn.AdaptiveAvgPool2d, AdaptiveAvgPool2d),
    LAYER_REGISTRY_ITEM(nn.BatchNorm2d, BatchNorm2d),
    LAYER_REGISTRY_ITEM(BinaryOp, BinaryOp),
    LAYER_REGISTRY_ITEM(torch.cat, Cat),
	...
};

// 例如：宏LAYER_REGISTRY_ITEM(DenseLayer, DenseLayer) 等价于 { "DenseLayer", { DenseLayer_LayerCreator, DenseLayer_LayerDestroyer } }
#define LAYER_REGISTRY_ITEM(pnnx_type, type)           \
    { #pnnx_type, { type##_LayerCreator, type##_LayerDestroyer } }
```

layer的creator方法和Destroyer方法，具体的定义由各个layer自己维护。通过定义宏，可以方便地定义各种类型的层（Layer）创建器函数，而不需要重复编写相似的代码。

```c++
DEFINE_LAYER_REGISTRY(ReLU);

#define DEFINE_LAYER_REGISTRY(type) \
    DEFINE_LAYER_CREATOR(type)      \
    DEFINE_LAYER_DESTROYER(type)

#define DEFINE_LAYER_CREATOR(type) \
    Layer* type##_LayerCreator() { \
        return (new type);         \
    }

#define DEFINE_LAYER_DESTROYER(type)           \
    void type##_LayerDestroyer(Layer* layer) { \
        if (nullptr != layer) {                \
            delete layer;                      \
        }                                      \
    }
```

然后是创建`layer`:

```c++
const LayerRegistryEntry* GetLayerRegistry(std::string type) {
    if (layer_registry_map.count(type) > 0) {
        return &layer_registry_map[type];
    }

    return nullptr;
}

const LayerRegistryEntry* layer_registry_entry = GetLayerRegistry(op->type);
Layer* layer = layer_registry_entry->creator();
```



### 线程池

创建线程池：

```c++
std::unique_ptr<Eigen::ThreadPool> eigen_threadpool_;
std::unique_ptr<Eigen::ThreadPoolDevice> eigen_threadpool_device_;

void Context::InitEigenThreadPoolDevice(int num_threads) {
    eigen_threadpool_.reset(new Eigen::ThreadPool(num_threads));
    eigen_threadpool_device_.reset(
        new Eigen::ThreadPoolDevice(eigen_threadpool_.get(), num_threads));
}
```

得到可用的线程：

```c++
Eigen::ThreadPoolDevice* Layer::GetEigenThreadPoolDevice() {
    if (nullptr != context_) {
        return context_->GetEigenThreadPoolDevice();
    }

    return Context::GetDefaultEigenThreadPoolDevice();
}

Eigen::ThreadPoolDevice* Context::GetEigenThreadPoolDevice() {
    return eigen_threadpool_device_.get();
}
```



### ReLU

直接使用`Eigen`提供的方法

```c++
Status ReLU::Forward(const Tensor& input, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    const EigenTensorMap<float, 1> input_eigen_tensor = input.GetEigenTensor<float, 1>();
    EigenTensorMap<float, 1> output_eigen_tensor = output.GetEigenTensor<float, 1>();

    // cwiseMax: 逐元素地计算两个矩阵或向量之间的最大值
    output_eigen_tensor.device(*device) = input_eigen_tensor.cwiseMax(0.0f);

    return Status::kSuccess;
}
```



### 卷积

仿佛在写python

```c++
    auto expr = input_eigen_tensor
                    .extract_image_patches(kernel_w_,
                                           kernel_h_,
                                           stride_w_,
                                           stride_h_,
                                           dilation_w_,
                                           dilation_h_,
                                           1,
                                           1,
                                           padding_t_,
                                           padding_b_,
                                           padding_l_,
                                           padding_r_,
                                           static_cast<float>(0))
                    .reshape(pre_contract_dims)
                    // 张量收缩是矩阵乘积对多维情况的推广, 可以实现高维的矩阵内积
                    .contract(kernel_tensor.reshape(kernel_dims), contract_dims)
                    .reshape(post_contract_dims);
```



## Engine

计算图的调度和执行，其中，`Eigen`类负责对外提供稳定的接口，`EngineImpl`类负责具体实现。

```c++
class EngineImpl;
class Engine {
public:
    Engine();
    ~Engine();

public:
    Status LoadModel(const std::string& parampath, const std::string& binpath);
    Status Release();

public:
    const std::vector<std::string> InputNames();
    const std::vector<std::string> OutputNames();

public:
    Status Input(const std::string& name, const Tensor& input);
    Status Forward();
    Status Extract(const std::string& name, Tensor& output);

private:
    EngineImpl* impl_ = nullptr;
};
```



### 创建计算图：`pnnx::Graph`

```c++
Status EngineImpl::CreateGraph(const std::string& parampath,
                               const std::string& binpath) {
    graph_  = new pnnx::Graph;
    int ret = graph_->load(parampath, binpath);
    if (0 != ret) {
        LOG(ERROR) << "load graph fail";
        return Status::kFail;
    }

    // 展开计算图中的表达式算子
    pnnx::expand_expression(*graph_);

    return Status::kSuccess;
}
```



### 创建张量节点：`TensorNode`

```c++
struct TensorNode {
    pnnx::Operand* operand = nullptr;
    Tensor tensor;
};

Status EngineImpl::CreateTensorNodes() {
    for (size_t i = 0; i < graph_->operands.size(); ++i) {
        pnnx::Operand* opd = graph_->operands[i];

        if (tensor_nodes_.count(opd->name) > 0) {
            LOG(ERROR) << "tensor node [" << opd->name << "] already exists";
            return Status::kFail;
        }

        TensorNode* tensor_node = new TensorNode;
        tensor_node->operand    = opd;

        // NCHW -> NHWC
        std::vector<int> shape_nhwc = opd->shape;
        if (shape_nhwc.size() > 3) {
            int shape_dims             = (int)shape_nhwc.size();
            shape_nhwc[shape_dims - 1] = opd->shape[shape_dims - 3];
            shape_nhwc[shape_dims - 2] = opd->shape[shape_dims - 1];
            shape_nhwc[shape_dims - 3] = opd->shape[shape_dims - 2];
        }

        tensor_node->tensor = Tensor(PnnxToDataType(opd->type), shape_nhwc, false);

        tensor_nodes_[opd->name] = tensor_node;

        if (nullptr != opd->producer) {
            // type == "pnnx.Input"
            if (0 == opd->producer->inputs.size()) {
                input_tensor_nodes_[opd->name] = tensor_node;
            }
        }

        for (auto& consumer : opd->consumers) {
            if (nullptr != consumer) {
                // type == "pnnx.Output"
                if (0 == consumer->outputs.size()) {
                    output_tensor_nodes_[opd->name] = tensor_node;
                    break;
                }
            }
        }
    }

    return Status::kSuccess;
}
```



### 创建计算节点：`Layer`

```c++
Status EngineImpl::CreateLayers() {
    for (size_t i = 0; i < graph_->ops.size(); ++i) {
        pnnx::Operator* op = graph_->ops[i];

        if ("pnnx.Input" == op->type || "pnnx.Output" == op->type) {
            continue;
        }

        if (layers_.count(op->name) > 0) {
            LOG(ERROR) << "layer [" << op->name << "] already exists";
            return Status::kFail;
        }

        const LayerRegistryEntry* layer_registry_entry =
            GetLayerRegistry(op->type);
        if (nullptr == layer_registry_entry) {
            LOG(ERROR) << "layer type [" << op->type << "] not registered";
            return Status::kEmpty;
        }

        Layer* layer = layer_registry_entry->creator();
        if (nullptr == layer) {
            LOG(ERROR) << "create layer [" << op->type << "] fail";
            return Status::kFail;
        }

        {
            Status ret = layer->Init(op);
            if (Status::kSuccess != ret) {
                LOG(ERROR) << "layer [" << op->name << "] init fail";
                return ret;
            }

            layer->SetContext(context_);
        }

        {
            std::vector<SimpleInfer::TensorNode*> input_tensor_nodes;
            for (size_t j = 0; j < op->inputs.size(); ++j) {
                pnnx::Operand* opd = op->inputs[j];
                if (tensor_nodes_.count(opd->name) > 0) {
                    input_tensor_nodes.push_back(tensor_nodes_[opd->name]);
                } else {
                    LOG(ERROR) << "tensor node [" << op->name << "] not exist";
                    return Status::kEmpty;
                }
            }

            layer->SetInputNodes(input_tensor_nodes);
        }

        {
            std::vector<SimpleInfer::TensorNode*> output_tensor_nodes;
            for (size_t j = 0; j < op->outputs.size(); ++j) {
                pnnx::Operand* opd = op->outputs[j];
                if (tensor_nodes_.count(opd->name) > 0) {
                    output_tensor_nodes.push_back(tensor_nodes_[opd->name]);
                } else {
                    LOG(ERROR) << "tensor node [" << op->name << "] not exist";
                    return Status::kEmpty;
                }
            }

            layer->SetOutputNodes(output_tensor_nodes);
        }

        {
            Status ret = layer->Validate();
            if (Status::kSuccess != ret) {
                LOG(ERROR) << "layer [" << op->name << "] validate fail";
                return ret;
            }
        }

        layers_[op->name] = layer;
    }

    return Status::kSuccess;
}
```



### 创建计算图执行流

```c++
Status EngineImpl::CreatePipeline() {
    pipeline_ = CGraph::GPipelineFactory::create();
    if (nullptr == pipeline_) {
        LOG(ERROR) << "create pipeline fail";
        return Status::kEmpty;
    }

    for (auto& layer_iter : layers_) {
        Layer* layer           = layer_iter.second;
        std::string layer_name = layer->GetOp()->name;

        if (pipeline_nodes_.count(layer_name) > 0) {
            LOG(ERROR) << "pipeline node [" << layer_name << "] already exists";
            return Status::kFail;
        }

        CGraph::GElementPtr element = nullptr;
        CStatus ret =
            pipeline_->registerGElement<PipelineNode>(&element, {}, layer_name);
        if (!ret.isOK() || nullptr == element) {
            LOG(ERROR) << "registerGElement fail";
            return Status::kFail;
        }

        PipelineNode* node = static_cast<PipelineNode*>(element);
        node->SetLayer(layer);
        node->setName(layer_name);
        pipeline_nodes_[layer_name] = node;
    }

    for (auto& tensor_node_iter : tensor_nodes_) {
        TensorNode* tensor_node = tensor_node_iter.second;

        pnnx::Operator* producer = tensor_node->operand->producer;
        if ("pnnx.Input" == producer->type) {
            continue;
        }

        std::string producer_name = producer->name;

        if (pipeline_nodes_.count(producer_name) <= 0) {
            LOG(ERROR) << "layer [" << producer_name << "] not in pipeline";
            return Status::kFail;
        }

        PipelineNode* producer_node = pipeline_nodes_[producer_name];
        CGraph::GElementPtrSet element_set{
            static_cast<CGraph::GElementPtr>(producer_node)};

        std::vector<pnnx::Operator*> consumers =
            tensor_node->operand->consumers;
        for (auto& consumer : consumers) {
            if ("pnnx.Output" == consumer->type) {
                continue;
            }

            std::string consumer_name = consumer->name;

            if (pipeline_nodes_.count(consumer_name) > 0) {
                PipelineNode* consumer_node = pipeline_nodes_[consumer_name];
                consumer_node->addDependGElements(element_set);
            } else {
                LOG(ERROR) << "layer [" << consumer_name << "] not in pipeline";
                return Status::kFail;
            }
        }
    }

    {
        CStatus ret = pipeline_->init();
        if (!ret.isOK()) {
            LOG(ERROR) << "pipeline init fail";
            return Status::kFail;
        }
    }

    return Status::kSuccess;
}
```



#### CGraph

`CGraph`是一个图流程执行框架，可以分析节点之间的依赖关系，并通过调度实现依赖元素之间按顺序执行，非依赖元素之间并发执行。
推理引擎的计算需求与图的执行流程基本一致，可以将每一个layer作为图中的一个可执行节点，通过模型的描述添加依赖关系。

对于分支结构比较多的网络可以并发计算那些互不依赖的layer。



### 计算图的执行

```c++
Status EngineImpl::Forward() {
    {
        CStatus ret = pipeline_->run();
        if (!ret.isOK()) {
            LOG(ERROR) << "pipeline run fail";
            return Status::kFail;
        }
    }

    return Status::kSuccess;
}
```





## 参考资料

- https://github.com/zpye/SimpleInfer
- [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)