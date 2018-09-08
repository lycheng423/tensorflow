import tensorflow as tf


# 通过 tf.variable_scope 和 tf.get_variable 函数
# 以下代码对 5.2.1 小节中定义的计算前向传播结果的函数做了一些改进。
# 不再需要将所有变量都作为参数传递到不同的函数中了。
def inference(input_tensor, reuse=False):
    # 定义第一层神经网络的变量和前向传播过程。
    with tf.variable_scope('layerl', reuse=reuse):
        # 根据传进来的reuse来判断是创建新变量还是使用己经创建好的。
        # 在第一次构造网络时需要创建新的变量，以后每次调用这个函数都直接使用reuse=True就不需要每次将变传进来了。
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_norinal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 类似地定义第二层神经网络的变量和前向传播过程。
    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    # 返回最后的前向传播结果。
    return layer2


x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
y = inference(x)

# 在程序中需要使用训练好的神经网络进行推导时，可以直接调用inference(new_x,True)
# 如果需要使用滑动平均模型可以参考 5.2.1 小节中使用的代码，把计算滑动平运的类传到inference函数中即可。
# 获取或者创建变量的部分不需要改变。
new_x = '...'
new_y = inference(new_x, True)
