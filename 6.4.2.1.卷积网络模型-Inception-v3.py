import tensorflow as tf

# 在 LeNet-5 模型中，不同卷积层通过串 联的方式连接在一起，
# 而 Inception-v3 模型中的 Inception 结构是将不同的卷积层通过并联 的方式结合在一起。
# 在下面的篇幅中将具体介绍 Inception 结构，并通过 TensorFlow-Slim 工具来实现 Inception-v3 模型中的一个模块
# Inception-v3 模型总共有 46 层，由 11 个 Inception 模块组成。
# 图 6-17 中方框标注出来 的结构就是一个 Inception 模块。
# 在 Inception-v3 模型中有 96 个卷积层，如果将 6.4.1 小节 中的程序直接搬过来，
# 那么一个卷积层就需要 5 行代码，于是总共需要 480 行代码来实现 所有的卷积层。
# 这样使得代码的可读性非常差。为了更好地实现类似 InCepti0 n-v3 模型这样的复杂卷积神经网络，
# 在下面将先介绍 TensorFlow-Slim 工具来更加简洁地实现一个卷积层。
# 以下代码对比了直接使用 TensorFlow 实现一个卷积层和使用 TensorFlow-Slim 实现同 样结构的神经网络的代码量。

# 直接使用TensorFlow 原始API实现卷积层。
with tf.variable_scope(scope_name):
    weights = tf.get_variable("weight", ...)
    biases = tf.get_variable('bias', ...)
    conv = tf.nn.conv2d(...)
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases))

# 使用TensorFlow-Slim实现卷积层
# 通过TensorFlow-Slim可以在一行中实现一个卷积层的前向传播算法。
# slim.conv2d 函数的有 3 个参数是必填的。
# 第一个参数为输入节点矩阵，第二参数是当前卷积层过滤器的深度，第三个参数是过滤器的尺寸。
# 可选的参数有过滤器移动的步长、是否使用全 0 填充、激活函数的选择以及变量的命名空间等。
net = slim.conv2d(input, 32, [3, 3])

# 加载 slim 库
slim = tf.contrib.slim

# slim.arg_scope 函数可以用于设置默认的参数取值。
# slim.arg_scope 函数的第一个参数是一个函数列表，在这个列表中的函数将使用默认的参数取值。
# 比如通过下面的定义，调用slim.conv2d(net, 320, [1, 1])函数时会自动加上stride=1和padding='SAME'的参数。
# 如果在函数调用时指定了stride, 那么这里设置的默认值就不会再使用。通过这种方式可以进一步减少冗余的代码。

with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
    ...
    # 此处省略了InCePtion-v3模型中其他的网络结构而直接实现最后面红色方框中的Inception结构。
    # 假设输入图片经过之前的神经网络前向传播的结果保存在变量net中
    net = 上一层的输出节点矩阵
    # 为一个Inception模块声明一个统一的变量命名空间
    with tf.variable_scope('Mixed_7c'):
        # 给 Inception 模块中每一条路径声明一个命名空间。
        with tf.variable_scope('Branch_0'):
            # 实现一个过滤边长为 1, 深度为320的卷积层。
            branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')

        # Inception模块中第二条路径。这条计算路径上的结构本身也是一个Inception结构
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
            # tf.concat函数可以将多个矩阵拼接起来。tf.concat函数的第一个参数指定了拼接的维度，
            # 这里给出的“3”代表了矩阵是在深度这个维度上进行拼接
            branch_1 = tf.concat(3, [
                # 此处 2 层卷积层的输入都是 branch_1 而不是 net
                slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')
            ])

        # Inception模块中第三条路径。此计算路径也是一个Inception结构。
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
            branch_2 = tf.concat(3, [
                slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')
            ])

        # Inceptiori模块中第四条路径。
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(
                net, [3, 3], scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(
                branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

        # 当前 Inception 模块的最后输出是由上面四个计算结果拼接得到的。
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])