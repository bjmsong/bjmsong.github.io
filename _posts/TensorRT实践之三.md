## PyTorch 对 ONNX 的算子支持

在转换`torch.nn.Module`模型时，PyTorch 一方面会用跟踪法执行前向推理，把遇到的算子整合成计算图；另一方面，PyTorch 还会把遇到的每个算子翻译成 ONNX 中定义的算子。在这个翻译过程中，可能会碰到以下情况：

- 该算子可以一对一地翻译成一个 ONNX 算子。
- 该算子在 ONNX 中没有直接对应的算子，会翻译成一至多个 ONNX 算子。
- 该算子没有定义翻译成 ONNX 的规则，报错。

```
RuntimeError: ONNX export failed: Couldn't export operator foo
```

我们需要了解ONNX支持哪些算子，可以在官方的[算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md)中查看。

![1682412768774](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1682412768774.png)

这份文档中最重要的是开头这个算子变更表格。表格的第一列是算子名，第二列是该算子发生变动的算子集版本号，也就是我们之前在`torch.onnx.export`中提到的`opset_version`表示的算子集版本号。通过查看算子第一次发生变动的版本号，我们可以知道某个算子是从哪个版本开始支持的；通过查看某算子小于等于`opset_version`的第一个改动记录，我们可以知道当前算子集版本中该算子的定义规则。

我们需要了解PyTorch对ONNX算子的映射，[ONNX SUPPORTED TORCHSCRIPT OPERATORS](https://pytorch.org/docs/stable/onnx_supported_aten_ops.html)

在 PyTorch 中，和 ONNX 有关的定义全部放在 [torch.onnx 目录](https://github.com/pytorch/pytorch/tree/master/torch/onnx)中，如下图所示：

![1682412985074](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1682412985074.png)

其中，`symbloic_opset{n}.py`（符号表文件）即表示 PyTorch 在支持第 n 版 ONNX 算子集时新加入的内容。

例如， bicubic 插值是在第 11 个版本开始支持的。我们以它为例来看看如何查找算子的映射情况。每一个`g.op`就是一个 ONNX 的定义。比如其中的 `Resize` 算子就是这样写的：

```python
g.op("Resize",
     input,
     empty_roi,
     empty_scales,
     output_size,
     coordinate_transformation_mode_s=coordinate_transformation_mode,
     cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
     mode_s=interpolate_mode,  # nearest, linear, or cubic
     nearest_mode_s="floor")  # only valid when mode="nearest"
```

通过在前面提到的 ONNX 算子文档中查找 [Resize 算子的定义](https://github.com/onnx/onnx/blob/main/docs/Operators.md#resize)，我们就可以知道这每一个参数的含义了。用类似的方法，我们可以去查询其他 ONNX 算子的参数含义，进而知道 PyTorch 中的参数是怎样一步一步传入到每个 ONNX 算子中的。 掌握了如何查询 PyTorch 映射到 ONNX 的关系后，我们在实际应用时就可以在 `torch.onnx.export()`的`opset_version`中先预设一个版本号，碰到了问题就去对应的 PyTorch 符号表文件里去查。如果某算子确实不存在，或者算子的映射关系不满足我们的要求，我们就可能得用其他的算子绕过去，或者自定义算子。





## 自定义算子

要使 PyTorch 算子顺利转换到 ONNX ，我们需要保证以下三个环节都不出错：

- 算子在 PyTorch 中有实现
- 有把该 PyTorch 算子映射成一个或多个 ONNX 算子的方法
- ONNX 有相应的算子

可在实际部署中，这三部分的内容都可能有所缺失。但所谓车到山前必有路，对于这三个环节，我们也分别都有以下的添加支持的方法：

- PyTorch 算子
  - 组合现有算子
  - 添加 TorchScript 算子
  - 添加普通 C++ 拓展算子
- 映射方法
  - 为 ATen 算子添加符号函数
  - 为 TorchScript 算子添加符号函数
  - 封装成 torch.autograd.Function 并添加符号函数
- ONNX 算子
  - 使用现有 ONNX 算子
  - 定义新 ONNX 算子



### 支持 ATen 算子

> [ATen](https://pytorch.org/cppdocs/#aten) 是 PyTorch 内置的 C++ 张量计算库，PyTorch 算子在底层绝大多数都是用 ATen 实现的。

如果算子在 ATen 中已经实现了，ONNX 中也有相关算子的定义，但是算子映射成 ONNX 的规则没有写。在这种情况下，我们只需要**为 ATen 算子补充描述映射规则的符号函数**就行了。





### 支持 TorchScript 算子





## 解决【TensorRT实践之二】提到的支持动态模型的问题

PyTorch的`interpolate`算子会映射到ONNX的`Resize`算子，但是`Resize`算子的输入`scales`是一个常量，我们希望可以支持`scales`为变量。可以自己实现新的`interpolate`算子。其中，`forward()`方法定义了算子的具体实现：

```python
class NewInterpolate(torch.autograd.Function):

    @staticmethod
    def symbolic(g, input, scales):
        # scales: [1, 1, w, h]
        return g.op("Resize",
                    input,
                    g.op("Constant",
                         value_t=torch.tensor([], dtype=torch.float32)),
                    scales,
                    coordinate_transformation_mode_s="pytorch_half_pixel",
                    cubic_coeff_a_f=-0.75,
                    mode_s='cubic',
                    nearest_mode_s="floor")

    @staticmethod
    def forward(ctx, input, scales):
        scales = scales.tolist()[-2:]
        return interpolate(input,
                           scale_factor=scales,
                           mode='bicubic',
                           align_corners=False)
    
    
class StrangeSuperResolutionNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        x = NewInterpolate.apply(x, upscale_factor)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out
```

`symbolic()`方法把这个算子映射到对应的ONNX算子上，其中`g.op` 的每个参数都可以映射到 ONNX 中的算子属性：

![1682408375521](C:\Users\宋伟清\AppData\Roaming\Typora\typora-user-images\1682408375521.png)







## 参考资料

- https://mmdeploy.readthedocs.io/zh_CN/dev-1.x/tutorial/01_introduction_to_model_deployment.html

- https://pytorch.org/docs/stable/onnx.html