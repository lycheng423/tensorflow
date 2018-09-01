import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = tf.add(a, b, name="add")
print(result)
# 张量：Tensor("add:0", shape=(2,), dtype=float32)
# 名字（节点名称：第几个节点的输出）， 纬度（元组）， 类型
