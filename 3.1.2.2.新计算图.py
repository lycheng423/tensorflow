import tensorflow as tf

# tf.Graph可以生成新计算图
g1 = tf.Graph()
# 在 计算图g1中 定义变量v，并设置初始值为0
with g1.as_default():
    v = tf.get_variable("v", initializer=tf.zeros_initializer()(shape=[2, 3]))

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[3, 4]))

# 在 计算图 g2中取定义变量v的值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))
# [[0. 0. 0.]
#  [0. 0. 0.]]

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))
# [[1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]]
