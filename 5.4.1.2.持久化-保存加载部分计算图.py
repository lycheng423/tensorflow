import tensorflow as tf

#有时可能只需要保存或者加载部分变量。比如，可能有一个之前训练好的五层神经网络模型, 但现在想尝试一个六层的神经网络，那么可以将前面五层神经网络中的参数直接加载到新 的模型，而仅仅将最后一层神经网络重新训练。

# 声明两个变量并计算他们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1], name="v1"))
v2 = tf.Variable(tf.constant(2.0, shape=[1], name="v2"))
result = v1 + v2

# 声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

# 保存模型
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 将模型保存到/data/tf/model.ckpt
    saver.save(sess, '/data/tf/model.ckpt')
    # model.ckpt.meta 保存计算图的结构，这里可以简单理解为神经网络的网络结构
    # model.ckpt.data-00000-of-00001 保存了 TensorFlow 程序中每一个变量的取值
    # checkpoint 保存了一个目录下所有的模型文件列表
    # model.ckpt.index
    # TensorFlow会将计算图的结构和图上的参数取值分开保存

# 加载模型
# 这段加载模型的代码基本上和保存模型的代码是一样的。
# 在加载模型的程序中也是先定义了 TensorFlow 计算图上的所有运算，并声明了一个 tf.train.Saver 类。
# 两段代码唯一不同的是，在加载模型的代码中没有运行变量的初始化过程，而是将变量的值通过己经保存的模型加载进来。
with tf.Session() as sess:
    # 加载己经保存的模型，并通过己经保存的模型中变跫的值来计算加法。
    saver.restore(sess, "/data/tf/model.ckpt")
    print(sess.run(result))
    # 输出[ 3.]

# 如果不希望重复定义图上的运算，也可以直接加载己经持久化的图。以下代码给出了一个样例。
# 直接加载持久化的图。
saver = tf.train.import_meta_graph("/data/tf/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "/data/tf/model.ckpt")
    # 通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
    # 输出[ 3.]