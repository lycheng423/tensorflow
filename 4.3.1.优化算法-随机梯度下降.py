import tensorflow as tf

# 神经网络的优化过程可以分为两个阶段
# 第一个阶段先通过前向传播算法计算得到预测值，并将预测值和真实值做对比得出两者之间的差距。
# 然后在第二个阶段通过反向传播算法计算损失函数对每一个参数的梯度，再根据梯度和学习率使用梯度下降算法更新每一个参数。

# 为了综合梯度下降算法和随机梯度下降算法的优缺点，在实际应用中一般釆用这两个算法的折中-每次计算一小部分训练数据的损失函数。这一小部分数据被称之为一个 batch
batch_size = n

# 每次读取一小部分数据作为当前的训练数据来执行反向传播算法
x = tf.placeholder(tf.float32, shape=(batch_size, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y-input')

# 定义神经网络结构和优化算法
loss = ''
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    # 参数初始化

    # 迭代更新参数
    for i in range(STEPS):
        # 准备batch_size个训练数据。一般所有训练数据随即打乱后再选取可以得到更好的优化效果
        current_X, current_Y = ...
        sess.run(train_step, feed_dict={x: current_X, y_: current_Y})
