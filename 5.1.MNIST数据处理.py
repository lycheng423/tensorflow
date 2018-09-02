from tensorflow.examples.tutorials.mnist import input_data

# 训练集与测试集数据的下载
# 训练集图片 - 55000 张 训练图片, 5000 张 验证图片
# http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# 训练集图片对应的数字标签
# http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# 测试集图片 - 10000 张 图片
# http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
# 测试集图片对应的数字标签
# http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# 载入MNIST数据集，如果指定地址/path/to/MNIST_data下没有已经下载好的数据，
# 那么TensorFlow 会自动从上面的网址下载数据
mnist = input_data.read_data_sets("/data/MNIST_data/", one_hot=True)

# 数据集	                目的
# data_sets.train	    55000组图片和标签, 用于训练。
# data_sets.validation	5000组图片和标签, 用于迭代验证训练的准确性。
# data_sets.test	    10000组图片和标签, 用于最终测试训练的准确性。

# 打印 Training data size: 55000
print("Training数据集:", mnist.train.num_examples)

# 打印 Validating data size: 5000
print("Validating数据集:", mnist.validation.num_examples)

# 打印 Testing data size: 10000
print("Testing数据集:", mnist.test.num_examples)

# 打印 Example training data: [0. 0. 0. ... 0.380 0.376 ... 0. ]
print('Example training data:', mnist.train.images[0])

# 打印 Example training data label: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print('Example training data label:', mnist.train.labels[0])

# 处理后的每一张图片是一个长度为 28x28=784 的一维数组，这个数组中的元素对应了图片像素矩阵中的每一个数字。
# 像素矩阵中元素的取值范围为[0,1]，它代表了颜色的深浅。其中 0 表示白色背景 ( background)，1 表示黑色前景(foreground )。

# 为了方便使用随机梯度下降，input_data.read_data_sets 函数生成的类还提供了mnist.train.next_batch 函数，它可以从所有的训练数据中读取一小部分作为一个训练 batch
batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)  # 从train的集合中选取batch_size个训练数据。
print("X shape:", xs.shape)
# 输出X shape:(100, 784)
print("Y shape:", ys.shape)
# 输出Y shape:(100, 10)
