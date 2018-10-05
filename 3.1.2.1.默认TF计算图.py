import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")  # 定义常量
b = tf.constant([2.0, 3.0], name="b")
result = a + b
# sess = tf.Session()
# sess.run(result)
print(a.graph is tf.get_default_graph())  # True
# get_default_graph 获取当前默认的计算图
# a.grap 查看张量所属的计算图，未指定则为默认图
# 所以 a.grap == tf.get_default_graph()
