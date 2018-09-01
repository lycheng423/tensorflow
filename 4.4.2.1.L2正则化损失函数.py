import tensorflow as tf

# 过拟合指的是当一个模型过为复杂之后，它可以很好地“记忆”每一个训练数据中随机噪音的部分而忘记了要去“学习”训练数据中通用的趋势
# 为了避免过拟合问题，一个非常常用的方法是正则化,就是在损失函数中加入刻画模型复杂程度的指标。
# 假设用于刻画模型在训练数据上表现的损失函数为J(θ)，那么在优化时不是直接优化J(θ)，而是优化J(θ)+λR(w)
# 其中R(w)刻画的是模型的复杂程度，而λ表示模型复杂损失在总损失中的比例
# 注意这里θ表示的是一个神经网络中所有的参数，它包括边上的权重w和偏置项b

# 一般来说模型复杂度只由权重w决定
# 常用的刻画模型复杂度的函数R(w)有两种，
# 一种是L1正则化，计算公式是：R(w) = ‖w‖
# 另一种是L2正则化，计算公式是：R(w) = ‖w‖²
# 无论是哪一种正则化方式，基本的思想都是希望通过限制权重的大小，使得模型不能任意拟合训练数据中的随机噪音。

# 但这两种正则化的方法也有很大的区别。首先，L1正则化会让参数变得更稀疏，而L2正则化不会。
# 所谓参数变得更稀疏是指会有更多的参数变为0, 这样可以达到类似特征选取的功能。
# 之所以L2正则化不会让参数变得稀疏的原因是当参数很小时，比如0.001，这个参数的平方基本上就可以忽略了，于是模型不会进一步将这个参数调整为0。
# 其次L1正则化的计算公式不可导，而L2正则化公式可导。因为在优化时需要计算损失函数的偏导数，所以对含有L2正则化损失函数的优化要更加简洁。而且优化方法也有很多种。在实践中，也可以将L1 正则化和 L2 正则化同时使用:R(w) = ∑α‖wᵢ‖+(1-α)wᵢ²
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w)

# loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(λ)(w)
# 在上面的程序中，loss 为定义的损失函数，它由两个部分组成。
# 第一个部分是均方误差损失函数，它刻画了模型在训练数据上的表现
# 第二个部分就是正则化，它防止模型过度模拟训练数据中的随机噪音
# λ参数表示了正则化项的权重，也就是公式J(θ)+λR(w)中的λ。w为需要计算正则化损失的参数。

# tf.contrib.layers.l2_regularizer函数可以返回一个函数，这个函数可以计算一个给定参数的L2正则化项的值。
# tf.contrib.layers.l1_regularizer可以计算L1正则化项的值。

weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
    # 输出为 (1+|-2|+|-3|+4)*0.5 = 5
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    # 输出为 (1²+|-2|²+|-3|²+4²)/2*0.5 = 7.5
    # TensorFlow 会将 L2 的正则化损失值除以 2 使得求导得到的结果更加简洁
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))
