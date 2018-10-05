import tensorflow as tf

# tf提供了通过变量名称来创建或者获取一个变量的机制。
# 通过这个机制，在不同的函数中可以直接通过变量的名字来使用变量，而不需要将变量通过参数的形式到处传递
# 下面这两个定义是等价的。
v = tf.get_variable("v", shape=[1], initializer=tf.constant.initializer(1.0))
v = tf.Variable(tf.constant(1.0, shape=[1]), name="v")

# 7 种不同的初始化函数

# 初始化函数                             功能                          主要参数
# tf.constant_initializer           将变量初始化为给定常量             常量的取值
# tf.random_normal_initializer      将变量初始化为满足正态分布的随机值   正态分布的均值和标准差
# tf.truncated_normal_initializer   将变量初始化为满足正态分布的随机值，
#                                   但如果随机出来的值偏离平均值超过差2个标准差，那么这个数将会被重新随机
# tf.random_uniform_initializer     将变量初始化为满足平均分布的随机值   最大、最小值
# tf.uniform_unit_scaling_initializer 将变量初始化为满足平均分布但     factor(产生随机值时乘以的系数)
#                                      不影响输出数量级的随机值
# tf.zeros_initializer              将变量设置为全 0                 变量维度
# tf.ones_initializer                将变量设置为全 1                 变量维度
