# Pytorch

## 入门教程
* 初始化张量
    * torch.empty(size)
    * torch.rand(size)      // (0, 1)均匀抽取随机数
    * torch.randn(size)     // 均值0、方差1的高斯分布
    * torch.zeros(size, dtype)
    * torch.tensor([1,1])
    * x.new_ones(size, dtype)
    * torch.randn_like(x, dtype)
* 张量基本操作
    * 数学运算，比如
        * x + y
        * torch.add(x, y, out=result)
        * y.add_(x)     // 直接改变发起运算的y的值
    * x([:, 1])     // 和numpy类似的截取
    * x.view(size)      // 改变形状
    * x如果只有一个元素，那么 x.item() 获取这个元素的值
    * 张量a, a.numpy() 转换为np数组
    * np数组a, torch.from_numpy(a) 转换为张量
    * .to()，移动存储位置到 gpu/cpu
    * .backward()，计算导数
* autograd
    * autograd 模块实现前向传播的求导，自动计算张量操作的微分
    * 设置张量的属性 .requires_grad = True，将开启 autograd 功能，跟踪对张量的所有操作，调用 .detach() 可以关闭功能、停止跟踪
    * 在 with torch.no_grad() 内，临时禁止求导，常在使用测试集评估时用到
    * 对张量的计算完成后，调用 .backward() 就能获得自动计算的梯度
        * 执行 z.backward()，获取 z.grad_fn，遍历里面的函数，以及函数 .next_functions 属性里的函数，递归直到叶节点，叶节点 .variable 指向的就是结果变量 z 的创建变量，计算结果就存储在创建变量的 .grad 属性中；求导结束，所有叶节点的 .grad 都得到了更新。
* 函数 Function
    * 张量和函数相互连接，建立一个无环图，该图记录了完整的计算历史，并进行编码；
    * 张量的 .grad_fn 属性，就引用了创建该张量的函数，如果是用户创建的，那么该属性为 None；
    * 结果张量（非手动创建的张量，计算得出），.grad_fn.next_functions，指向创建运算的前一步运算，反复查找 .next_functions，会找到计算图的叶节点，类型为 AcuumulateGrad，其属性 .variable 指向叶节点。
* NN
    * 只需要定义 forward 函数，后向传播的 backward 会自动计算；
    * torch.nn 只支持小批量梯度下降 mini-batches，nn.Conv2d() 需要输入一个四阶张量，nSamples x nChannels x Height x Width


## 实践
### 简单图像分类
* 数据集是 CIFAR10，共包含10类物体，图像的尺寸为 32 x 32 x 3；
* 采取一个简单的CNN, 2 x CONV + 3 x FC
1. 加载数据集
    * 指定对图像进行的变换：
    >
        torchvision.transforms.Compose(transforms)
        * .ToTensor()，将 PIL Image / numpy.ndarray 转换为张量 Tensor
        * .Normalize(mean, std, inplace=False)，归一化，指定均值和标准差，注意如果有n个通道，那么需要指定为 (mean[1], ..., mean[n]), (std[1], ..., std[n])
    * 下载数据集：
    >
        torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
        * root，数据集存储的根目录；
        * train=True，读取的是训练集数据，否则是测试集的；
        * transfrom 是对图像的变换操作，按上面的操作可以结合多种操作；
        * target_transform 是对真实值的操作；
        * download=True 表示需要下载数据集。
    * 加载数据集：
    >
        torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
        * dataset，加载的数据集
        * batch_size，批的大小
        * shuffle=True 则每轮训练 epoch 都会打乱数据
        * num_workers，加载数据同时使用的进程数

2. 定义一个CNN网络
    * 定义一个类 Net(nn.Module)
    * 初始化 \_\_init\_\_()，定义网络结构
    * 卷积层
    >
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                dilation, groups, bias, padding_mode)
    * 最大池化层
    >
        nn.MaxPool2d(kernel_size, stride, padding,
                    dilation, return_indices, ceil_mode)
    * 全连接层
    >
        nn.Linear(in_features, out_features, bias)
    * 前向传播 forward()，组装定义的各层网络
        * 注意激活函数，比如ReLU，需调用 torch.nn.functional.relu()
        * 卷积基部分结束以后，需要用 view() 展开张量

3. 设置训练参数
    * 设置损失函数，比如 nn.CrossEntropyLoss()
    * 设置优化器，比如 torch.SGD(params, lr, momentum)
    * 设置训练轮数 epochs

4. 训练常规参数
    * 开始 epoch 循环：
        * 枚举 trainloader，获取当前的 inputs, labels
        * 清空梯度缓存，optimizer.zero_grad()
        * 模型中输入 inputs，得到预测结果 ouputs
        * 根据预测结果和真实结果，计算损失函数 loss
        * 反向传播：loss.backward()
        * 更新参数：optimizer.step()
    * 保存训练好的模型，调用 torch.save(state_dict, path)
    * 加载保存的模型，调用 .load_state_dict(torch.load(path))

5. 在测试集上测试模型效果
    * 注意使用 with torch_no_grad()，暂时关闭 autograd 功能
    * 加载 testloader，获得当前的 images, labels
    * 应用训练好的模型，获取预测结果 outputs，对应各个类别的概率
    >
        torch.max(input, dim, keepdim=False, out=None)
            returns (values, indices)
    * 调用 torch.max() 获取最大概率分类的索引，也就是预测的分类，同真实分类作比较，计算准确率

6. 在 gpu 上训练时
    * 在 gpu 上建立模型，net.to(device)
    * 每次读取的 inputs, labels 也都需要调用 .to() 发送到 gpu 上