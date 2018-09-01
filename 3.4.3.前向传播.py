import tensorflow as tf

######## 第一步：定义计算 ########
# 声明两个变量
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 输入的特征向量，矩阵
x = tf.constant([
    [0.7, 0.9]
])

# 矩阵乘法，前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

######## 第二步：通过会话运行计算 ########
sess = tf.Session()

# 一个变量值在被使用前，变量的初始化要被明确的调用
# 1.直接调用每个变量初始化过程
# sess.run(w1.initializer)  # 初始化w1
# sess.run(w2.initializer)  # 初始化w2
# 2.初始化所有变量
init_op = tf.global_variables_initializer()
sess.run(init_op)

# 打印结果 [[3.957578]]
print(sess.run(y))

sess.close()
