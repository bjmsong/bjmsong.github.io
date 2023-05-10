## PyTorch模型转ONNX

```python
import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
model = torchvision.models.alexnet(pretrained=True).cuda()

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
```



### Tracing vs Scripting

`torch.onnx.export`中需要的模型是`torch.jit.ScriptModule`。而要把`torch.nn.Module`转换为 `torch.jit.ScriptModule` 模型，有`tracing`和`scripting`两种导出计算图的方法。

如果给`torch.onnx.export`传入`torch.nn.Module`，会默认使用`tracing`的方法导出。

所谓`tracing`：即给定一组输入，实际执行一遍模型，把这组输入对应的计算图记录下来，保存为 ONNX 格式。

如果模型是动态，即包含控制流（分支语句、循环语句），这种方式无法捕捉到所有的动态情况，需要通过`scripting`的方式导出模型。

![1682408688096](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1682408688096.png)



### 参数讲解

[API](https://pytorch.org/docs/stable/onnx.html#functions)

```python
def export(model, args, f, export_params=True, verbose=False, training=TrainingMode.EVAL,
           input_names=None, output_names=None, aten=False, export_raw_ir=False,
           operator_export_type=None, opset_version=None, _retain_param_name=True,
           do_constant_folding=True, example_outputs=None, strip_doc_string=True,
           dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None,
           enable_onnx_checker=True, use_external_data_format=False):
```



#### input_names, output_names

很多推理引擎在运行 ONNX 文件时，都需要以“名称-张量值”的数据对应输入数据，并根据输出张量的名称来获取输出数据。在进行跟张量有关的设置（比如添加动态维度）时，也需要知道张量的名字。 在实际的部署流水线中，我们都需要设置输入和输出张量的名称，并保证 ONNX 和推理引擎中使用同一套名称。



#### dynamic_axes

指定输入输出张量的哪些维度是动态的。 为了追求效率，ONNX 默认所有参与运算的张量都是静态的（张量的形状不发生改变）。但在实际应用中，我们又希望模型的输入张量是动态的，尤其是没有形状限制的卷积模型。因此，我们需要显式地指明输入输出张量的哪几个维度是可变的。 我们来看一个例子：

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        return x


model = Model()
dummy_input = torch.rand(1, 3, 10, 10)
model_names = ['model_static.onnx', 'model_dynamic_0.onnx', 'model_dynamic_23.onnx']


dynamic_axes_0 = {
    'in' : [0],
    'out' : [0]
}

"""
由于 ONNX 要求每个动态维度都有一个名字，这样写的话会引出一条 UserWarning，警告我们通过列表的方式设置动态维度的话系统会自动为它们分配名字。一种显式添加动态维度名字的方法如下：
dynamic_axes_0 = {
    'in' : {0: 'batch'},
    'out' : {0: 'batch'}
}
"""

dynamic_axes_23 = {
    'in' : [2, 3],
    'out' : [2, 3]
}

# 没有动态维度
torch.onnx.export(model, dummy_input, model_names[0],
input_names=['in'], output_names=['out'])
# 第0维动态
torch.onnx.export(model, dummy_input, model_names[1],
input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_0)
# 第2第3维动态
torch.onnx.export(model, dummy_input, model_names[2],
input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_23)
```





### 避开陷阱

#### 不要使用`Numpy`和Python内置类型

通过`tracing`方式导出的模型，会把`numpy`和Python内置类型当作常量，引起错误。使用`torch.Tensor`可以避免这个问题。

```python
# Bad! Will be replaced with constants during tracing.
x, y = np.random.rand(1, 2), np.random.rand(1, 2)
np.concatenate((x, y), axis=1)

# Good! Tensor operations will be captured during tracing.
x, y = torch.randn(1, 2), torch.randn(1, 2)
torch.cat((x, y), dim=1)

# Bad! y.item() converts a Tensor to a Python built-in number
def forward(self, x, y):
    return x.reshape(y.item(), -1)

# Good! y will be preserved as a variable during tracing.
def forward(self, x, y):
    return x.reshape(y, -1)
```



#### 不要使用`Tensor.data`

 用[`torch.Tensor.detach()`](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html#torch.Tensor.detach)替代



#### 避免in-place操作

下面的例子中`real_seq_length`和`seq_length`共享同一份内存，会带来错误。要改成`real_seq_length = real_seq_length + 2`

```python
class Model(torch.nn.Module):
  def forward(self, states):
      batch_size, seq_length = states.shape[:2]
      real_seq_length = seq_length
      real_seq_length += 2
      return real_seq_length + seq_length
```



### 局限性

- `Scripting`模式，不支持部分对`tuple`和`list`的操作，比如`appending a tuple to a list`

- 不支持的`Tensor`索引方式

  ```python
  # Tensor indices that includes negative values.
  data[torch.tensor([[1, 2], [2, -3]]), torch.tensor([-2, 3])]
  # Workarounds: use positive index values.
  ```

  ```python
  # Multiple tensor indices if any has rank >= 2
  data[torch.tensor([[1, 2], [2, 3]]), torch.tensor([2, 3])] = new_data
  # Workarounds: use single tensor index with rank >= 2,
  #              or multiple consecutive tensor indices with rank == 1.
  
  # Multiple tensor indices that are not consecutive
  data[torch.tensor([2, 3]), :, torch.tensor([1, 2])] = new_data
  # Workarounds: transpose `data` such that tensor indices are consecutive.
  
  # Tensor indices that includes negative values.
  data[torch.tensor([1, -2]), torch.tensor([-2, 3])] = new_data
  # Workarounds: use positive index values.
  
  # Implicit broadcasting required for new_data.
  data[torch.tensor([[0, 2], [1, 1]]), 1:3] = new_data
  # Workarounds: expand new_data explicitly.
  # Example:
  #   data shape: [3, 4, 5]
  #   new_data shape: [5]
  #   expected new_data shape after broadcasting: [2, 2, 2, 5]
  ```

  

### 支持动态模型

为了让模型的泛用性更强，在推理阶段有更大的自由度，需要在尽可能不影响原有逻辑的前提下，让模型的输入输出或是结构动态化。

以下面的超分辨模型为例，图片的放大比例是写死在模型里的（`upscale_factor=3`）：

```python
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4)
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)

    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

model = init_torch_model()
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "srcnn.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])
```

假设我们要做一个超分辨率的应用，希望图片的放大倍数能够自由设置。而我们交给用户的，只有一个 .onnx 文件和运行超分辨率模型的应用程序。

可以使用 `interpolate` 代替 `nn.Upsample`，令模型的放大倍数变成推理时的输入：

```python
class SuperResolutionNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        x = interpolate(x,
                        scale_factor=upscale_factor,
                        mode='bicubic',
                        align_corners=False)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model():
    torch_model = SuperResolutionNet()

    # Please read the code about downloading 'srcnn.pth' and 'face.png' in
    # https://mmdeploy.readthedocs.io/zh_CN/1.x/tutorial/01_introduction_to_model_deployment.html#pytorch
    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


model = init_torch_model()

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(model, (x, 3),
                      "srcnn2.onnx",
                      opset_version=11,
                      input_names=['input', 'factor'],
                      output_names=['output'])
```

这里有个问题，我们传入的第二个参数” 3 “是一个整形变量。这不符合上面【避开陷阱】里面的规定。我们可以修改模型的输入。

```python
...

class SuperResolutionNet(nn.Module):

    def forward(self, x, upscale_factor):
        x = interpolate(x,
                        scale_factor=upscale_factor.item(),
                        mode='bicubic',
                        align_corners=False)
...

with torch.no_grad():
    torch.onnx.export(model, (x, torch.tensor(3)),
                      "srcnn2.onnx",
                      opset_version=11,
                      input_names=['input', 'factor'],
                      output_names=['output'])
```

但是，导出 ONNX 时却报了一条 `TraceWarning` 的警告，说有一些量可能会追踪失败。打开ONNX模型可以发现，虽然我们把模型推理的输入设置为了两个，但 ONNX 模型还是长得和原来一模一样，只有一个叫 ” input ” 的输入。

![1682407071938](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1682407071938.png)

这是由于我们使用了 `torch.Tensor.item()` 把数据从 Tensor 里取出来，而导出 ONNX 模型时这个操作是无法被记录的，只好报了一条 `TraceWarning`。这导致 interpolate 插值函数的放大倍数还是被设置成了” 3 “这个固定值。

仔细观察导出的ONNX模型，发现无论是使用最早的 `nn.Upsample`，还是后来的`interpolate`，PyTorch 里的插值算子最后都会转换成 ONNX 定义的 Resize 算子。也就是说，所谓 PyTorch 转 ONNX，实际上就是把每个 PyTorch 的算子映射成了 ONNX 定义的算子。

点击该算子，可以看到它的详细参数如下：

![1682407415395](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1682407415395.png)

其中，展开 scales，可以看到 scales 是一个长度为 4 的一维张量，其值为 [1, 1, 3, 3], 表示 Resize 操作每一个维度的缩放系数；其类型为 Initializer，表示这个值是根据常量直接初始化出来的。如果我们能够自己生成一个 ONNX 的 Resize 算子，让 scales 成为一个可变量而不是常量，就像它上面的 X 一样，那这个超分辨率模型就能动态缩放了。

因此，我们需要自定义算子。



### 实战技巧

#### 使模型在 ONNX 转换时有不同的行为

有些时候，我们希望模型在直接用 PyTorch 推理时有一套逻辑，而在导出的ONNX模型中有另一套逻辑。比如，我们可以把一些后处理的逻辑放在模型里，以简化除运行模型之外的其他代码。`torch.onnx.is_in_onnx_export()`可以实现这一目标，该函数仅在执行 `torch.onnx.export()`时为真。以下是一个例子：

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        if torch.onnx.is_in_onnx_export():
            x = torch.clip(x, 0, 1)
        return x
```

这里，我们仅在模型导出时把输出张量的数值限制在[0, 1]之间。



#### 利用中断张量跟踪的操作

PyTorch 转 ONNX 的`tracing`导出方式不是万能的。如果我们在模型中做了一些很“出格”的操作，`tracing`法会把某些取决于输入的中间结果变成常量，从而使导出的ONNX模型和原来的模型有出入。以下是一个会造成这种“跟踪中断”的例子：

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * x[0].item()
        return x, torch.Tensor([i for i in x])

model = Model()
dummy_input = torch.rand(10)
torch.onnx.export(model, dummy_input, 'a.onnx')
```

如果你尝试去导出这个模型，会得到一大堆 warning，告诉你转换出来的模型可能不正确。这也难怪，我们在这个模型里使用了`.item()`把 torch 中的张量转换成了普通的 Python 变量，还尝试遍历 torch 张量，并用一个列表新建一个 torch 张量。这些涉及张量与普通变量转换的逻辑都会导致最终的 ONNX 模型不太正确。 另一方面，我们也可以利用这个性质，在保证正确性的前提下令模型的中间结果变成常量。这个技巧常常用于模型的静态化上，即令模型中所有的张量形状都变成常量。



### 使用[**onnx-simplifier**](https://github.com/daquexian/onnx-simplifier)优化ONNX模型

```shell
pip install -U pip && pip install onnxsim
onnxsim input_onnx_model output_onnx_model
```

如果遇到报错[`ModuleNotFoundError: No module named 'onnxruntime'`](https://github.com/microsoft/onnxruntime/issues/11907)

```shell
pip uninstall onnxruntime onnxruntime-gpu
```



### 直接下载训练好的ONNX

https://github.com/onnx/models



## 参考资料

- https://mmdeploy.readthedocs.io/zh_CN/dev-1.x/tutorial/01_introduction_to_model_deployment.html
- https://pytorch.org/docs/stable/onnx.html
- https://onnx.ai/onnx/intro/index.html