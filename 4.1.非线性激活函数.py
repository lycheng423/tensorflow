import tensorflow as tf

# 加入 非线性激活函数 和 偏置项 的 神经网络 的 向前传播算法
a = tf.nn.relu(tf.matmul(x, w1)) + bisases1)
y = tf.nn.relu(tf.matmul(a, w2)) + bisases2)
# 常用的非线性激活函数：
# 1.ReLU激活函数(tf.nn.relu)：f(x) = max(x,0)
# 2.sigmoid激活函数(tf.sigmoid)：f(x) = 1/(1+e(-x次幂))
# 3.tanh激活函数(tf.tanh)：f(x) = (1-e(-2x次幂))/(1+e(-2x次幂))

# bisases偏置项，可以表达为一个输出永远为1的节点
