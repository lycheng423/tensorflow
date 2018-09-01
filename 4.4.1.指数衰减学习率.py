import tensorflow as tf

global_step = tf.Variable(0)

# 学习率不能过大，也不能过小。TensorFlow提供了一种更加灵活的学习率设置方法-指数衰减法。
# tf.train.exponential_decay函数实现了指数衰减学习率，会指数级地减小学习率
# 通过这个函数，先使用较大的学习率快速得到一个比较优的解，然后随机迭代的继续逐步减小学习率
learning_rate = tf.train.exponential_decay(
    0.1, global_step, 100, 0.96, staircase=True
)
# learning_rate=0.1 初始学习率
# decay_steps=100 衰减速度，迭代轮数
# decay_rate=0.96 衰减系数
# staircase 默认false，学习率随迭代轮数变化的趋势为曲线。true为阶梯状（global_step/decay_steps为整数）
# 由于指定了staircase=True，所以每次训练100轮后学习率*0.96
# 一般来说初始学习率、衰减系数和衰减速度都是根据经验设置
# 损失函数下降速度和迭代结束之后总损失大小没有必然联系，即并不能通过前几轮损失函数下降的速度来比较不同神经网络的效果

# 实现了以下代码的功能
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
# decayed_learning_rate 每轮优化时使用的学习率
# learning_rate 事先设定的初始学习率
# decay_rate 衰减系数
# decay_steps 衰减速度


# 通过exponential_decay函数生成学习率


# 使用指数衰减的学习率。
# 在minimize函数中传入global_step将自动更新global_step参数，从而使得学习率也得到相应更新
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
