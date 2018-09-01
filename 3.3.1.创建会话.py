import tensorflow as tf

# 1
# 创建一个会话
sess = tf.Session()
sess.run(result)
sess.close()

# 2 自动关闭
with tf.Session() as sess:
    sess.run()
