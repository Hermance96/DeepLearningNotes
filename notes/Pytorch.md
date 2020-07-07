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
    * x如果只有一个元素，那么 x.item() 获取内部元素的值
    * 张量a, a.numpy() 转换为np数组
    * np数组a, torch.from_numpy(a) 转换为张量
    * .to()，移动存储位置到 gpu/cpu
    * .backward()，计算导数
* autograd 包
    * 设置张量的属性，.requires_grad 为 True，将跟踪对张量的所有操作，完成计算后，调用.backward() 就能获得自动计算的所有梯度；
    * 张量的梯度都记录到 .grad 属性中，调用 .detach() 可以停止跟踪张量计算。
    * 张量和函数相互连接并建立一个无环图，该图对完整的计算历史进行编码；
    * 张量的 .grad_fn 属性，引用创建了张量的函数（用户创建的张量该属性为 None）;
