# Tesnsorflow
## 数据集
### tensorflow_datasets
* tfds.load()
    * name，数据集名称；
    * download=True, 默认先调用 tfds.core.DatasetBuilder.download_and_prepare 下载数据；
    * as_supervise=False, 默认返回所有特征的 dictionary，若为真，则返回 (input, label) 二元组；
    * with_info=False, 若改为真，返回元组 (tf.data.Dataset, tfds.core.DatasetInfo)

### tf.data.Dataset
1. 创建数据集
* from_tensor_slices([python list])
* TextLineDataset([txt flie name])
* list_files('/path/*.txt')
* from_generator(generator, output_types, output_shapes)
2. 应用变换，处理数据
* map(map_func, num_parallel_calls); 对数据集的每个元素都应用 map_func，返回一个新数据集，num_parallel_calls 是并行处理的元素个数，默认为按顺序依次处理；
* func = ds.filter(); ds.apply(func); 过滤数组元素；
* as_numpy_iterator()，返回一个迭代器，将 dataset 元素都转换为 numpy，用于查看数据集的具体内容；查看数据集的数据类型和形状，直接 print() 元素即可。
* batch(batch_size, drop_remainder=False); 将数据集的连续元素合并为 batch，如果不能整除批的大小，默认保留最后一个批，
* cache(filename); 缓存数据集中的元素，第一次迭代时，数据集的元素被缓存在指定文件/内存中；
* shuffle(buffer_size); 选取 buffer_size 大小的元素，然后随机从 buffer 取出元素，替代原数据集；
3. 在数据集上小批量迭代

## 图像处理 tf.image
* resize(images, size, method); 调整图片大小
* crop(); 图片剪裁
* flip_left_right(); flip_up_down(); 左右/上下翻转

## Eager
* 操作立刻返回具体的值，而不是先构建计算图，然后再运行静态图
* 接口更直观，可使用 Python 数据结构；调试简单；使用 Python 控制流，简化动态模型；支持大多数tf操作和GPU加速。
* 在 tf2 中，默认开启 eager 模式
    * 查看是否处于 eager 模式： tf.executing_eagerly()
* tf.Tensor 和 ndarray
    * 调用方法 Tensor.numpy() 即可
    * 也可以直接使用 np 方法操作 Tensor，得到结果为 ndarray
    * tf.math 将 Python 对象、ndarray 转换为 Tensor
* 自动求导在神经网络模型的反向传播中用的比较多，tf.GradientTape 用于跟踪梯度计算
    * 所有前向传播的计算都相当于放在一个 “tape” 上，反向传播计算梯度时，相当于做倒带，然后计算完丢弃磁带
    * 对于变量 Variable 自动追踪，对于张量 Tensor，需要调用 tape.watch() 方法
    > 
        with tf.GradientTape() as tape:
             // forward operations
        grads = tape.gradient(losses, training_variables)
* 例：基于数据集 MNIST 训练一个简单模型
1. 加载和处理数据集
    * tf.keras.datasets.mnist，留下训练集的 images, labels
    * 从训练集的图片和标签，调用方法 from_tensor_slices，建立 tf.data.DataSet
    * 调整数据类型，图片归一化至 [0, 1]
    * 数据集进行 shuffle & batch
2. 建立一个简单的2卷积层，Sequential模型
    * 定义模型优化器和损失函数
3. 定义一个训练步骤 train_step(images, labels)
    * 开启梯度追踪，计算模型前向传播的输出，以及根据损失函数计算损失
    * 计算梯度，tape.gradient(losses, training_vars)
    * 应用优化器更新参数，optimizer.apply_gradient(zip(grads, training_vars))
4. 开始训练
    * 对于每一轮训练
    * 枚举数据集 -> batch, (images, labels)
    * 对于每个batch，调用一次训练 train_step
5. 训练结束，绘图/打印训练结果

## Variable
* 变量和张量互相转换
    > 
        tf.Variable(a_tansor, name='')
        tf.convert_to_tensor(a_var)
* 常用属性：形状 shape，数据类型 dtype
* 常用方法：
    * 转换为 ndarray: v.numpy()
    * 重新赋值：v.assign(new_tensor); v.assign_add(add_tensor);
    * 改变形状：tf.reshape(v, shape)
    * 求最大值的索引：tf.argmax(v, axis=0); 默认axis=0，求列的最大值索引

## Tensor
* 张量初始化： ts = tensor.constant([1])    // 1阶张量
* 常用方法：
    * 转换为 ndarray: np.asarray(ts); ts.numpy();
    * 张量运算：tf.add(a, b); tf.multiply(a, b); tf.matmul(a, b);
    * 用符号表达为：a+b; a*b; a@b;
    * 求所有元素的最大值： tf.reduce_max(ts)