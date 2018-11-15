# pytorch 

## jupyter notebook Tutorial
IPython notebook

## Tensor
a tensor can be considered to be high order array.
firstly:
`import torch as t`

1. construct a tensor 
```python
x = torch.Tensor(5,3) // have not init
x = t.Tensor([[1,2],[3,4]])
x = t.rand(5, 3) // random
```
x.size()[i] or x.size(i)
to view the i demension's size

for example 
x.size()[1] = x.size(1) = 3;
x.size()[0] = x.size(0) = 5;


2. add two tensor
```python
y = t.rand(5,3)

x + y #1

t.add(x+y) #2

result = t.Tensor(5,3) #3
t.add(x, y, out=result)

print('最初y')
print(y)

print('第一种加法，y的结果')
y.add(x) # 普通加法，不改变y的内容
print(y)

print('第二种加法，y的结果')
y.add_(x) # inplace 加法，y变了
print(y)

```
NOTE:函数名后面带下划线_ 的函数会修改Tensor本身。例如，x.add_(y)和x.t_()会改变 x，但x.add(y)和x.t()返回一个新的Tensor， 而x不变。
The function with an underscore _ after the function name modifies the Tensor itself. For example, x.add_(y) and x.t_() change x, but x.add(y) and x.t() return a new Tensor, and x does not change.

/# Tensor的选取操作与Numpy类似
`x[:, 1] # select the i column`

### tensor and numpy
Interoperability between arrays of Tensor and Numpy is very easy and fast. For operations that are not supported by Tensor, you can first convert to Numpy array processing and then back to Tensor.

Tensor和Numpy的数组之间的互操作非常容易且快速。对于Tensor不支持的操作，可以先转为Numpy数组处理，之后再转回Tensor。

```python
a = t.ones(5)
\# a tensor([ 1.,  1.,  1.,  1.,  1.])

b = a.numpy() # Tensor -> Numpy

c = np.ones(5)
d = c.form_numpy(c) # numpy ->tensor
#Tensor和numpy对象共享内存，所以他们之间的转换很快，而且几乎不会消耗什么资源。但这也意味着，如果其中一个变了，另外一个也会随之改变。

d.add_(1)# 以`_`结尾的函数会修改自身
print(c)
print(d) # Tensor和Numpy共享内存

```

如果你想获取某一个元素的值，可以使用scalar.item。 直接tensor[idx]得到的还是一个tensor: 一个0-dim 的tensor，一般称为scalar.
```python
In [25]:

scalar = b[0]

scalar

Out[25]:

tensor(2., dtype=torch.float64)

In [26]:

scalar.size() #0-dim

Out[26]:

torch.Size([])

In [27]:

scalar.item() # 使用scalar.item()能从中取出python对象的数值

Out[27]:

2.0

In [28]:

tensor = t.tensor([2]) # 注意和scalar的区别

tensor,scalar

Out[28]:

(tensor([2]), tensor(2., dtype=torch.float64))

In [29]:

tensor.size(),scalar.size()

Out[29]:

(torch.Size([1]), torch.Size([]))

In [30]:

/# 只有一个元素的tensor也可以调用`tensor.item()`

tensor.item(), scalar.item()

Out[30]:

(2, 2.0)

此外在pytorch中还有一个和np.array 很类似的接口: torch.tensor, 二者的使用十分类似。
In [31]:

tensor = t.tensor([3,4]) # 新建一个包含 3，4 两个元素的tensor

In [32]:

scalar = t.tensor(3)

scalar

Out[32]:

tensor(3)

In [33]:

old_tensor = tensor

new_tensor = t.tensor(old_tensor)

new_tensor[0] = 1111

old_tensor, new_tensor

Out[33]:

(tensor([3, 4]), tensor([1111,    4]))

需要注意的是，t.tensor()总是会进行数据拷贝，新tensor和原来的数据不再共享内存。所以如果你想共享内存的话，建议使用torch.from_numpy()或者tensor.detach()来新建一个tensor, 二者共享内存。
In [34]:

new_tensor = old_tensor.detach()

new_tensor[0] = 1111

old_tensor, new_tensor

Out[34]:

(tensor([1111,    4]), tensor([1111,    4]))
```


## GPU Tensor
```python
/# 在不支持CUDA的机器下，下一步还是在CPU上运行
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
z = x+y
```

## autograd

tensor can use autograd since torchv 0.4.
suggest use tensor replace the older one Variable.

`tensor.requries_grad = True`

example:

```python

\# 为tensor设置 requires_grad 标识，代表着需要求导数
\# pytorch 会自动调用autograd 记录操作
x = t.ones(2, 2, requires_grad=True)
\# 上一步等价于
/# x = t.ones(2,2)
/# x.requires_grad = True
x

y = x.sum()
/# tensor(4., grad_fn=<SumBackward0>)

y.grad_fn
/# <SumBackward0 at 0x7f9993a2b710>

y.backward() # 反向传播,计算梯度
# y = x.sum() = (x[0][0] + x[0][1] + x[1][0] + x[1][1])
# 每个值的梯度都为1
x.grad 

#tensor([[1., 1.],
#       [1., 1.]])

#grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。\
# 以下划线结束的函数是inplace操作，会修改自身的值，就像add_
x.grad.data.zero_() 

```

## Neural Network

Autograd实现了反向传播功能，但是直接用来写深度学习的代码在很多情况下还是稍显复杂，torch.nn是专门为神经网络设计的模块化接口。nn构建于 Autograd之上，可用来定义和运行神经网络。nn.Module是nn中最重要的类，可把它看成是一个网络的封装，包含网络各层定义以及forward方法，调用forward(input)方法，可返回前向传播的结果。下面就以最早的卷积神经网络：LeNet为例，来看看如何用nn.Module实现。LeNet的网络结构如图2-7所示。


这是一个基础的前向传播(feed-forward)网络: 接收输入，经过层层传递运算，得到输出。
定义网络

定义网络时，需要继承nn.Module，并实现它的forward方法，把网络中具有可学习参数的层放在构造函数__init__中。如果某一层(如ReLU)不具有可学习的参数，则既可以放在构造函数中，也可以不放，但建议不放在其中，而在forward中使用nn.functional代替。

```py
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()
        
        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5) 
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5) 
        # 仿射层/全连接层，y = Wx + b
        self.fc1   = nn.Linear(16*5*5, 120) 
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        # 卷积 -> 激活 -> 池化 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x

net = Net()
print(net)

```

Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)

只要在nn.Module的子类中定义了forward函数，backward函数就会自动被实现(利用`autograd`)。在`forward` 函数中可使用任何tensor支持的函数，还可以使用if、for循环、print、log等Python语法，写法和标准的Python写法一致。

网络的可学习参数通过`net.parameters()`返回，`net.named_parameters`可同时返回可学习的参数及名称。


```
params = list(net.parameters())
print(len(params))
10


for name,parameters in net.named_parameters():

    print(name,':',parameters.size())
```
conv1.weight : torch.Size([6, 1, 5, 5])
conv1.bias : torch.Size([6])
conv2.weight : torch.Size([16, 6, 5, 5])
conv2.bias : torch.Size([16])
fc1.weight : torch.Size([120, 400])
fc1.bias : torch.Size([120])
fc2.weight : torch.Size([84, 120])
fc2.bias : torch.Size([84])
fc3.weight : torch.Size([10, 84])
fc3.bias : torch.Size([10])
```
forward函数的输入和输出都是Tensor。

```
input = t.randn(1, 1, 32, 32)
out = net(input)
out.size()
```
`torch.Size([1, 10])`

需要注意的是，torch.nn只支持mini-batches，不支持一次只输入一个样本，即一次必须是一个batch。但如果只想输入一个样本，则用 `input.unsqueeze(0)`将batch_size设为１。例如 `nn.Conv2d` 输入必须是4维的，形如$nSamples \times nChannels \times Height \times Width$。可将nSample设为1，即$1 \times nChannels \times Height \times Width$。



