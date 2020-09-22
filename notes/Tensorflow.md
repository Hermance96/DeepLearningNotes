# Tensorflow 1.0
* tf基础框架
    * 应用层：提供机器学习相关的训练库、预测库，实现对计算图的构造；
    * 接口层：对tf功能模块的封装，提供调用接口；
    * 核心层，包括：
        * 设备层，支持tf在不同硬件设备上的实现，向上提供统一的接口，实现程序的跨平台功能；
        * 网络层，主要包括RPC和RDMA通信协议，实现不同设备之间的数据传输和更新；
        * 数据操作层，处理张量 tensor，进行张量运算；
        * 图计算层，实现计算图的创建、编译、优化和执行等。

<img src="imgs/cp_graph1.png" width="500">

* 计算图，把张量和计算操作作为图的节点，通过有向边连接实现张量的流动。
    * 重点在于，图的定义和图的计算时完全分开的；
    * 需要预先定义各种变量，建立相关的数据流图，在数据流图中创建各种变量之间的计算关系，完成计算图的定义；
    * 然后创建会话 session 对象，在会话中传入输入数据，进行计算，得到输出值。
* 计算流程：
    * 创建 Tensor 变量；
    * 决定变量之间进行的运算；
    * 初始化张量；
    * 创建会话 Session 对象；
    * 在会话中调用 run() 运行计算图。
    * 如果在会话中传入张量的值，需要用 placeholder 初始化，在session中计算的时候，再指定feed_dict传入具体的值，比如 feed_dict={x: 2}
* 基本概念：
    * Session，会话，计算图的具体执行者，与计算图进行交互，一个会话可以包含多个图
    * Tensor，张量，tf中最主要的数据结构，张量的类型包括：
        * Constant，常量，创建常量节点，不会再修改它的值；
            * tf.constant(value, dtype, shape, name='', verify_shape=False)
        * Variable，变量，表示图中的各个计算参数，通过调整这些变量来优化机器学习算法
            * tf.Variable(<initial_value>, name='')
            * 变量的使用必须初始化
            * init = tf.global_variables_initializer()
            * 然后在会话中先执行 session.run(init)
    * Placeholder，占位符，声明数据位置，通过设定 feed_data 传入指定类型和形状的数据，在计算图运行时用获取的数据进行计算，计算完毕后获取的数据就会消失
        * tf.placeholder(dtype, shape, name)
    * Queue，队列，图中有状态的节点，入列返回计算图中的一个操作节点，出列返回一个张量值
    * Operation，操作，是图中的节点，输入和输出都是张量，包括：
        * 初始化，tf.ones(shape), tf.zeros(shape)
        * onehot编码处理，tf.one_hot(labels, depth, axis)
        * 数学运算
            * add, multiply, matmul
            * tf.reduce_mean()，相当于 np.mean()，求均值
        * NN 相关的计算：
        * 激活函数，tf.nn.relu()，tf.nn.sigmoid()
        * 求损失函数，传入最后一层得到的Z[L]和Y即可：
            * tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)

* 【例】用TF建立3层NN，处理多类别图像识别
* 加载数据并进行数据处理：
    - 加载数据集train_signs, test_signs，识别手比划的数字0到5
        - x_train_org, (1080, 64, 64, 3)，共180\*6个样本，每张照片宽和高为64，3个通道
        - y_train_org, (1, 1080), 每个训练集的样本对应1个结果，数字0到5
        - x_test_org, (120, 64, 64, 3)，共20\*6个样本，y_test_org, (1, 120)
    - 对x进行展开，然后除以255，缩小数据范围
        - x_train (12288, 1080), x_test (12288, 120)
    - 对y进行one-hot编码，扩展其维度
        - y_train (6, 1080), y_test (6, 120)
* 初始化参数，由于创建的是3层网络，所以需要对 W1, W2, W3随机初始化，对b1, b2, b3赋值为0初始化
    * 这里用 get_variable(name, shape, initializer) 创建变量
    * W用的初始器是 tf.contrib.layers.xavier_initializer(seed)
    * b用的初始器是 tf.zeros_initializer()
* 计算FP，输出Z3
* 计算损失函数，输出 cost
* 对于后向传播，创建优化器，指定学习率，明确目标为最小化 cost
* 应用小批量优化方法：
    * 训练轮数for循环，对于当前一轮
    * 根据 batch_size 决定批总数，随机分批，然后对于每一批进行for循环
    * sess.run([optimizer, cost], feed_dict)，计算当前批的损失
    * 将批的损失/批总数，计入当前轮的损失中
* 最后保存训练后的参数，作为返回值。分别再训练集和测试集上，评估模型准确率。

# Tesnsorflow 2.0
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
    * with tf.GradientTape() as tape: // forward operations
    * grads = tape.gradient(losses, training_variables)
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