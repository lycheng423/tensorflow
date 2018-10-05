import tensorflow as tf

# 有时可能只需要保存或者加载部分变量。
# 比如，可能有一个之前训练好的五层神经网络模型, 但现在想尝试一个六层的神经网络，
# 那么可以将前面五层神经网络中的参数直接加载到新的模型，而仅仅将最后一层神经网络重新训练。

# 为了保存或者加载部分变量，在声明 tf.train.Saver 类时可以提供一个列表来指定需要保存或加载的变量
# 比如在加载模型的代码中使用 saver = tf.train.Saver([v1])命令来构建 tf.tmin.Saver 类，
# 那么只有变量 v1 会被加载进来。如果运行修改后只加载了 v1 的代码会得到变量未初始化的错误:
# tensorflow.python.framework.errors.FailedPreconditionError: Attempting to use uninitialized value v2
# 因为 v2 没有被加载，所以 v2 在运行初始化之前是没有值的。

# 变量重命名
# 除了可以选取需要被加载的变量，tf.tmin.Saver 类也支持在保存或者加载时给变量重命名。
# 这里声明的变量名称和已经保存的模型中变量的名称不同。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")

# 如果直接使用 tf.train.Saver()来加载模型会报变量找不到的错误。
# 下面显示了报错信息: tensorflow.python.framework.errors.NotFoundError: Tensor name "other-v2"
#  not found in checkpoint files /data/tf/model.ckpr

# 使用一个字典(dictionary)来重命名变量可以就可以加载原来的模型了。
# 这个字典指定了原来名称为v1的变量现在加载到变量v1中(名称为other-v1),
# 名称为v2的变量，加载到变量v2中(名称为other-v2)
saver = tf.train.Saver({"v1": v1, "v2": v2})

#在这个程序中，对变量 v1 和 v2 的名称进行了修改。
# 如果直接通过 tf.train.Saver 默认的构造函数来加载保存的模型，那么程序会报变量找不到的错误。
# 因为保存时候变量的名称和加载时变量的名称不一致。
# 为了解决这个问题，TensorFlow 可以通过字典(dictionary) 将模型保存时的变量名和需要加载的变量联系起来。
# 这样做主要目的之一是方便使用变量的滑动平均值。