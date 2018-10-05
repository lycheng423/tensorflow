import tensorflow as tf

# 在 4.4.3 小节中介绍了使用变量的滑动平均值可以让神经网络模型更加健壮(robust)。
# 在 TensorFlow 中，每一个变量的滑动平均值是通过影子变量维护的，
# 所以要获取变量的滑动平均值实际上就是获取这个影子变量的取值。
# 如果在加载模型时直接将影子变量映射到变量自身，那么在使用训练好的模型时就不需要再调用函数来获取变量的滑动平均值了。
# 这样大大方便了滑动平均模型的使用。

v = tf.Variable(0, dtype=tf.float32, name="v")
# 在没有申明滑动平均模型时只有一个变量 v, 所以下面的语句只会输出 "v:0"
for variables in tf.global_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
# 在声明滑动平均模型后，TensorFlow会自动生成一个影子变量 v/ExponentialMoving Average。
# 于是下面的语句会输出 "v:0" 和 "v/ExponentialMovingAverage:0"
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存时，TensorFlow会将v:0和v/ExponentialMovingAverage:0两个变量都存下来。
    saver.save(sess, "/data/tf/model.ckpt")
    print(sess.run([v, ema.average(v)]))
    # 输出[10.0, 0.099999905]
