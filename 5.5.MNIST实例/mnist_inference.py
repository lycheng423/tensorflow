import tensorflow as tf

# 结合 5.3 节中介绍的变量管理机制和 5.4 节中介绍的 TensorFlow 模型持久化机制，
# 本节中将介绍一个 TensorFlow 训练神经网络模型的最佳实践。
# 将训练和测试分成两个独立的程序，这可以使得每一个组件更加灵活。
# 比如训练神经网络的程序可以持续输出训练好的模型，而测试程序可以每隔一段时间检验最新模型的正确率，如果模型效果更好，
# 则将这个模型提供给产品使用。除了将不同功能模块分幵，本节还将前向传播的过程抽象成一个单独的库函数。
# 因为神经网络的前向传播过程在训练和测试的过程中都会用到，所以通过 库函数的方式使用起来既可以更加方便，
# 又可以保证训练和测试过程中使用的前向传播方法一定是一致的。
# 本节将提供重构之后的程序来解决MNIST问题。重构之后的代码将会被拆成3个程序，
# 第一个是 mnist_inference.py, 它定义了前向传播的过程以及神经网络中的参数。
# 第二个是 mnist_train.py, 它定义了神经网络的训练过程。
# 第三个是 mnist_eval.py, 它定义了测试过程。
# 以下代码给出了 mnist_inference.py 中的内容。

# 定义神经网络结构相关的参数。
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 通过 tf.get_variable 函数来获取变量。
# 在训练神经网络时会创建这些变量；在测试时会通过保存的模型加载这些变量的取值。
# 而且更加方便的是，因为可以在变量加载时将滑动平均变量重命名，
# 所以可以直接通过同样的名字在训练时使用变量自身，而在测试时使用变量的滑动平均值。
# 在这个函数中也会将变量的正则化损失加入损失集合。
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    # 当给出了正则化生成函数时，将当前变量的正则化损失加入名字为losses的集合。
    # 在这里使用了add_to_collection函数将一个张量加入一个集合，而这个集合的名称为losses。
    # 这是自定义的集合，不在TensorFlow自动管理的集合列表中。
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义神经网络的前向传播过程。
def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变贵并完成前向传播过程
    with tf.variable_scope("layer1"):
        # 这里通过tf.get_variable或tf.Variable没有本质区别，
        # 因为在训练或是测试中没有在同一个程序年多次调用这个函数。
        # 如果在同一个程序中多次调用，在第一次调用之后需要将reuse参数设置为True。
        weights = get_weight_variable(
            [INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable(
            "biases", [LAYER1_NODE],
            initializer=tf.constant_initializer(0.0)
        )
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 类似的声明第二层神经网络的变量并完成前向传播过程。
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable(
            "biases", [OUTPUT_NODE],
            initializer=tf.constant_initializer(0.0)
        )
        layer2 = tf.matmul(layer1, weights) + biases

    # 返回最后前向传播的结果。
    return layer2

# 在这段代码中定了神经网络的前向传播算法。
# 无论是训练时还是测试时，都可以直接调用 inference 这个函数，而不用关心具体的神经网络结构。
