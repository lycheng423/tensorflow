import tensorflow as tf

# 为了方便加载时重命名滑动平均变量，tf.train.ExponentialMovingAverage 类提供了
# variables_to_restore 函数来生成tf.train.Saver类所需要的变量重命名字典。
# 以下代码给出了 variables_to_restore 函数的使用样例。
v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)
# 通过使用 variables_to_restore 函数可以直接生成上面代码中提供的字典{"v/ExponentialMovingAverage":v}
# 以下代码会输出:
# {'v/ExponentialMovingAverage': <tensorflow.python.ops.variables.Variable object at 0x7ff6454ddcl0>}
# 其中后面的Variable类就代表了变量v
print(ema.variables_to_restore())

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "/data/tf/model.ckpt")
    print(sess.run(v))
    # 输出 0.099999905, 即原来模型中变量 v 的滑动平均值。
