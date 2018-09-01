import tensorflow as tf

v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

sess = tf.InteractiveSession()
# greater比较两个张量中每个元素的大小
print(tf.greater(v1, v2).eval())  # [False False  True  True]

#where有三个参数，第一个为判断条件，为true时，会使用第二个参数的值，否则使用第三个参数
print(tf.where(tf.greater(v1, v2), v1, v2).eval())  # [4. 3. 3. 4.]

sess.close
