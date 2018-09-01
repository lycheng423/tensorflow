import tensorflow as tf

# 使用sigmoid函数将y转换为0～1之间的数值。
# 转换后y代表预测是正样本的概率。1-y代表预测是负样本的概率
y = tf.sigmoid(y)

# 损失函数刻画预测值与真实值的差距，即交叉熵，这是分类问题常用的损失函数
loss = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0))
)

# 定义学习率
learning_rate = 0.001
# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 比较常用的优化方法有：
# tf.train.GradientDescentOptimier
# tf.train.AdamOptimizer
# tf.train.MomentumOptimizer
