import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))

#定义一个位置，位置中的数据在运行时再指定
#数据类型需要指定，和张量一样，类型不可改变
x = tf.placeholder(tf.float32, shape=(1, 2), name="input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

init_op = tf.global_variables_initializer()
sess.run(init_op)

#传入数据
#feed_dict是个字典
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
