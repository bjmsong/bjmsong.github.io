## 模型构建

使用 TensorRT 构建模型主要有两种方式：

1. 直接通过 TensorRT 的 API （Python，C++）逐层搭建网络；
2. 将中间表示的模型转换成 TensorRT 的模型，比如将 ONNX 模型转换成 TensorRT 模型。



### 直接构建

利用 TensorRT 的 API 逐层搭建网络，这一过程类似使用 Pytorch 搭建网络。需要注意的是需要将权重内容加载到 TensorRT 的网络中。本文就不详细展示，只搭建一个对输入做池化的简单网络。



#### Python API

主要是利用 `tensorrt.Builder` 的 `create_builder_config` 和 `create_network` 功能，分别构建 config 和 network，前者用于设置网络的最大工作空间等参数，后者就是网络主体，需要对其逐层添加内容。

此外，需要定义好输入和输出名称，将构建好的网络序列化，保存成本地文件。值得注意的是：如果想要网络接受不同分辨率的输入输出，需要使用 `tensorrt.Builder` 的 `create_optimization_profile` 函数，并设置最小、最大的尺寸。

```python
import tensorrt as trt

verbose = True
IN_NAME = 'input'
OUT_NAME = 'output'
IN_H = 224
IN_W = 224
BATCH_SIZE = 1

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
with trt.Builder(TRT_LOGGER) as builder, 
     builder.create_builder_config() as config, 
     builder.create_network(EXPLICIT_BATCH) as network:
    # define network
    input_tensor = network.add_input(
        name=IN_NAME, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H, IN_W))
    pool = network.add_pooling(
        input=input_tensor, type=trt.PoolingType.MAX, window_size=(2, 2))
    pool.stride = (2, 2)
    pool.get_output(0).name = OUT_NAME
    network.mark_output(pool.get_output(0))

    # serialize the model to engine file
    profile = builder.create_optimization_profile()
    profile.set_shape_input('input', *[[BATCH_SIZE, 3, IN_H, IN_W]]*3)
    builder.max_batch_size = 1
    config.max_workspace_size = 1 << 30
    engine = builder.build_engine(network, config)
    with open('model_python_trt.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")
```



#### C++ API

整个流程和Python API的执行过程非常类似，需要注意的点有：

1. `nvinfer1:: createInferBuilder` 对应 Python 中的 `tensorrt.Builder`，需要传入 `ILogger` 类的实例，但是 `ILogger` 是一个抽象类，需要用户继承该类并实现内部的虚函数。不过此处我们直接使用了 TensorRT 包解压后的 samples 文件夹 ../samples/common/logger.h 文件里的实现 `Logger` 子类。
2. 设置 TensorRT 模型的输入尺寸，需要多次调用 `IOptimizationProfile` 的 `setDimensions` 方法，比 Python 略繁琐一些。`IOptimizationProfile` 需要用 `createOptimizationProfile` 函数，对应 Python 的 `create_builder_config` 函数。

```C++
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <../samples/common/logger.h>

using namespace nvinfer1;
using namespace sample;

const char* IN_NAME = "input";
const char* OUT_NAME = "output";
static const int IN_H = 224;
static const int IN_W = 224;
static const int BATCH_SIZE = 1;
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

int main(int argc, char** argv)
{
        // Create builder
        Logger m_logger;
        IBuilder* builder = createInferBuilder(m_logger);
        IBuilderConfig* config = builder->createBuilderConfig();

        // Create model to populate the network
        INetworkDefinition* network = builder->createNetworkV2(EXPLICIT_BATCH);
        ITensor* input_tensor = network->addInput(IN_NAME, DataType::kFLOAT, Dims4{ BATCH_SIZE, 3, IN_H, IN_W });
        IPoolingLayer* pool = network->addPoolingNd(*input_tensor, PoolingType::kMAX, DimsHW{ 2, 2 });
        pool->setStrideNd(DimsHW{ 2, 2 });
        pool->getOutput(0)->setName(OUT_NAME);
        network->markOutput(*pool->getOutput(0));

        // Build engine
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(IN_NAME, OptProfileSelector::kMIN, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
        profile->setDimensions(IN_NAME, OptProfileSelector::kOPT, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
        profile->setDimensions(IN_NAME, OptProfileSelector::kMAX, Dims4(BATCH_SIZE, 3, IN_H, IN_W));
        config->setMaxWorkspaceSize(1 << 20);
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        // Serialize the model to engine file
        IHostMemory* modelStream{ nullptr };
        assert(engine != nullptr);
        modelStream = engine->serialize();

        std::ofstream p("model.engine", std::ios::binary);
        if (!p) {
                std::cerr << "could not open output file to save model" << std::endl;
                return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        std::cout << "generating file done!" << std::endl;

        // Release resources
        modelStream->destroy();
        network->destroy();
        engine->destroy();
        builder->destroy();
        config->destroy();
        return 0;
}
```



#### [trtexec](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec)

命令行构建工具，主要有两个用途：

- 测试模型性能基准
- 模型构建

[trtexec配置项](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags)

参数解释

- `--workspace` : defaults to the full size of device global memory but can be restricted when necessary

- `--minShapes=<shapes>, --optShapes=<shapes>, and --maxShapes=<shapes>`:  Specify the range of the input shapes to build the engine with. Only required if the input model is in ONNX format.

- `--buildOnly`  : Exit after the engine has been built and skip inference perf measurement (default = disabled)

  

### IR转换模型

#### Python API

首先使用 Pytorch 实现一个和上文一致的模型，即只对输入做一次池化并输出；然后将 Pytorch 模型转换成 ONNX 模型；最后将 ONNX 模型转换成 TensorRT 模型。 这里主要使用了 TensorRT 的 `OnnxParser` 功能，它可以将 ONNX 模型解析到 TensorRT 的网络中。最后我们同样可以得到一个 TensorRT 模型，其功能与上述方式实现的模型功能一致。

```python
import torch
import onnx
import tensorrt as trt

onnx_model = 'model.onnx'

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)

device = torch.device('cuda:0')

# generate ONNX model
torch.onnx.export(NaiveModel(), torch.randn(1, 3, 224, 224), onnx_model, input_names=['input'], output_names=['output'], opset_version=11)
onnx_model = onnx.load(onnx_model)

# create builder and network
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

# parse onnx
parser = trt.OnnxParser(network, logger)

if not parser.parse(onnx_model.SerializeToString()):
    error_msgs = ''
    for error in range(parser.num_errors):
        error_msgs += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

config = builder.create_builder_config()
config.max_workspace_size = 1<<20
profile = builder.create_optimization_profile()

profile.set_shape('input', [1,3 ,224 ,224], [1,3,224, 224], [1,3 ,224 ,224])
config.add_optimization_profile(profile)
# create engine
with torch.cuda.device(device):
    engine = builder.build_engine(network, config)

with open('model.engine', mode='wb') as f:
    f.write(bytearray(engine.serialize()))
    print("generating file done!")
```

IR 转换时，如果有多 Batch、多输入、动态 shape 的需求，都可以通过多次调用 `set_shape` 函数进行设置。`set_shape` 函数接受的传参分别是：输入节点名称，可接受的最小输入尺寸，最优的输入尺寸，可接受的最大输入尺寸。一般要求这三个尺寸的大小关系为单调递增。



#### C++ API

通过 `NvOnnxParser`，可以将上一小节转换时得到的 ONNX 文件直接解析到网络中。

```c++
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <../samples/common/logger.h>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace sample;

int main(int argc, char** argv)
{
        // Create builder
        Logger m_logger;
        IBuilder* builder = createInferBuilder(m_logger);
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        IBuilderConfig* config = builder->createBuilderConfig();

        // Create model to populate the network
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

        // Parse ONNX file
        IParser* parser = nvonnxparser::createParser(*network, m_logger);
        bool parser_status = parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

        // Get the name of network input
        Dims dim = network->getInput(0)->getDimensions();
        if (dim.d[0] == -1)  // -1 means it is a dynamic model
        {
                const char* name = network->getInput(0)->getName();
                IOptimizationProfile* profile = builder->createOptimizationProfile();
                profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
                profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
                profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
                config->addOptimizationProfile(profile);
        }

        // Build engine
        config->setMaxWorkspaceSize(1 << 20);
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        // Serialize the model to engine file
        IHostMemory* modelStream{ nullptr };
        assert(engine != nullptr);
        modelStream = engine->serialize();

        std::ofstream p("model.engine", std::ios::binary);
        if (!p) {
                std::cerr << "could not open output file to save model" << std::endl;
                return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        std::cout << "generate file success!" << std::endl;

        // Release resources
        modelStream->destroy();
        network->destroy();
        engine->destroy();
        builder->destroy();
        config->destroy();
        return 0;
}
```



## 模型推理

### Python API

首先是使用 Python API 推理 TensorRT 模型，这里部分代码参考了 [MMDeploy](https://github.com/open-mmlab/mmdeploy)。运行下面代码，可以发现输入一个 `1x3x224x224` 的张量，输出一个 `1x3x112x112` 的张量，完全符合我们对输入池化后结果的预期。

```Python
from typing import Union, Optional, Sequence,Dict,Any

import torch
import tensorrt as trt

class TRTWrapper(torch.nn.Module):
    def __init__(self,engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs

model = TRTWrapper('model.engine', ['output'])
output = model(dict(input = torch.randn(1, 3, 224, 224).cuda()))
print(output)
```



### C++ API

```c++
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <../samples/common/logger.h>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;
using namespace sample;

const char* IN_NAME = "input";
const char* OUT_NAME = "output";
static const int IN_H = 224;
static const int IN_W = 224;
static const int BATCH_SIZE = 1;
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);


void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
        const ICudaEngine& engine = context.getEngine();

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine.getNbBindings() == 2);
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine.getBindingIndex(IN_NAME);
        const int outputIndex = engine.getBindingIndex(OUT_NAME);

        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 3 * IN_H * IN_W /4 * sizeof(float)));

        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
        // create a model using the API directly and serialize it to a stream
        char *trtModelStream{ nullptr };
        size_t size{ 0 };

        std::ifstream file("model.engine", std::ios::binary);
        if (file.good()) {
                file.seekg(0, file.end);
                size = file.tellg();
                file.seekg(0, file.beg);
                trtModelStream = new char[size];
                assert(trtModelStream);
                file.read(trtModelStream, size);
                file.close();
        }

        Logger m_logger;
        IRuntime* runtime = createInferRuntime(m_logger);
        assert(runtime != nullptr);
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
        assert(engine != nullptr);
        IExecutionContext* context = engine->createExecutionContext();
        assert(context != nullptr);

        // generate input data
        float data[BATCH_SIZE * 3 * IN_H * IN_W];
        for (int i = 0; i < BATCH_SIZE * 3 * IN_H * IN_W; i++)
                data[i] = 1;

        // Run inference
        float prob[BATCH_SIZE * 3 * IN_H * IN_W /4];
        doInference(*context, data, prob, BATCH_SIZE);

        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
}
```

![1682216388441](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1682216388441.png)





## 参考资料

- https://mmdeploy.readthedocs.io/zh_CN/dev-1.x/tutorial/06_introduction_to_tensorrt.html#id7
- https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html