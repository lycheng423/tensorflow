import tensorflow as tf

weights = tf.Variable(tf.random_nomal([2, 3], stddev=2))
# tf.Variable声明变量

# 随机数生成函数
# 产生2*3的矩阵，矩阵中的元素正态分布，标准差stddev=2，平均值默认mean=0
tf.random_nomal([2, 3], stddev=2)

# 常数生成函数
tf.zeros([2, 3], dtype=tf.int32)  # [[0,0,0],[0,0,0]]
tf.ones([2, 3], dtype=tf.int32)  # [[1,1,1],[1,1,1]]
tf.fill([2, 3], 9)  # [[9,9,9],[9,9,9]]
tf.constant([1, 2, 3])  # [1,2,3]

