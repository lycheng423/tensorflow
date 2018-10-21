import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
# 加载mnist_inference.py中定义的常量和前向传播的函数。
import mnist_inference

# 使用定义好的前向传播过程，以下代码给出了神经网络的训练程序 mnist_train.py

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "/data/tf_test/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(
        tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用mnist_inference.py中定义的前向传播过程。
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 和5.2.1小节样例中类似地定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')

    # 初始化 TensorFlow 持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run(
                [train_op, loss, global_step],
                feed_dict={x: xs , y_: ys}
            )

            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。
                # 通过损失函数的大小可以大概了解训练的情况。
                # 在验证数据集上的正确率信息会有一个单独的程序来生成。
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

                # 保存当前的模型。注意这里给出了 global_step 参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，
                # 比 “model.ckpt-1000” 表示训练 1000 轮之后得到的模型。
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("/data/tf_test", one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()

# 运行上面的程序，可以得到类似下面的结果。
# ~/mnist$ python mnist_train.py
# Extracting /tmp/data/train-images-idx3-ubyte.gz
# Extracting /tmp/data/train-labels-idxl-ubyte.gz
# Extracting /tmp/data/tl0k-images-idx3-ubyte.gz
# Extracting /tmp/data/tlOk-labels-idxl-ubyte.gz
# After 1 training step(s), loss on training batch is 3.32075.
# After 1001 training step(s), loss on training batch is 0.241039.
# After 2001 training step(s), loss on training batch is 0.227391.
# After 3001 training step(s), loss on training batch is 0.138462.
# After 4001 training step(s), loss on training batch is 0.132074.
# After 5001 training step(s), loss on training batch is 0.103472.
