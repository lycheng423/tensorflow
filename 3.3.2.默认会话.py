import tensorflow as tf

# tf不会自动生成默认的会话，需要指定
sess = tf.Session()
with sess.as_default():
    print(result.eval())  # eval计算张量的取值

# 等同于
sess = tf.Session()
# 1
print(sess.run(result))
# 2
print(result.eval(session=sess))
