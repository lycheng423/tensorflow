import tensorflow as tf

########分类问题：将不同的样本分到事先定义好的类别中########
# 交叉熵：两个概率分布之间的距离
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
)
# y_代表正确结果，y代表预测结果

# 拆分解析：
# 1.tf.clip_by_value将一个张量的数值限定在一个范围之内
v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(tf.clip_by_value(v, 2.5, 4.5).eval())
# 输出[[2.5 2.4 3.][4. 4.5 4.5]]
# 小于2.5的数被换成了2.5，而大于4.5的数都被换成4.5

# 2.tf.log对所有张量依次对数
v = tf.constant([1.0, 2.0, 3.0])
print(tf.log(v).eval())
# 输出[0. 0.69314718 1.09861231]

# 3.用*将两个元素直接相乘
v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
v2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
print((v1 * v2).eval())
# 输出[[5. 12.][21. 32.]]

# 4.reduce_mean对矩阵取平均值
v = tf.constant([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
print(tf.reduce_mean(v).eval())

# 交叉熵一般与softmax回归一起使用
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
# 得到使用了softmax回归之后的交叉熵

########回归问题：房价预测，销售预测########
# 解决回归问题的神经网络一般只有一个输出节点，这个节点的输出值就是预测值
# 对于回归问题最常用的是均方误差
# 用-将两个元素直接相减
mse = tf.resource_mean(tf.square(y_ - y))
