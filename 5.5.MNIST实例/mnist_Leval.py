import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 加载mnist_inference.py和mnist_train.py中定义的常量和函数。
import mnist_inference
import mnist_train
# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率。
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(
            tf.float32, [None, mnist_inference.INPUT_NODE], name = 'x-input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name = 'y_input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    # 直接通过调用封装好的函数来计算前向传播的结果。因为测试时不关注正则化损失的值,
    # 所以这里用于计算正则化损失的函数被设置为 None。
    y = mnist_inference.inference(x, None)

    # 使用前向传播的结果计算正确率。如果需要对未知的样例进行分类，
    # 那么使用tf.argmax(y, 1)就可以得到输入样例的预测类别了
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值了。
    # 这样就可以完全共用mnist_inference.py中定义的前向传播过程。
    variable_averages = tf.train.ExponentialMovingAverage(
        mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore=variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化。
    while True:
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新摄型的文件名。
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数。
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
                print("After %s training step(s), validation accuracy = %g" %
                      (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return
                time,sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("/data/tf_test", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()

#上面给出的mnist_eval.py程序会每隔10秒运行一次，
# 每次运行都是读取最新保存的模型，并在MNIST验证数据集上计算模型的正确率。
# 如果需要离线预测未知数据的类别(比如这个样例程序可以判断手写体数字图片中所包含的数字)，
# 只需要将计算正确率的部分改为答案输出即可。运行mniSt_eval.py程序可以得到类似下面的结果。
# 注意因为这个程序每10秒自动运行一次，而训练程序不一定每10秒输出一个新模型，
# 所以在下面的结果中会发现有些模型被测试了多次。一般在解决真实问题时，不会这么频繁地运行评测程序。
# ~/mnist$ python mnist_eval.py
# Extracting /tmp/data/train-images-idx3-ubyte.gz
# Extracting /tmp/data/train-labels-idxl-ubyte.gz
# Extracting /tmp/data/tl0k-images-idx3-ubyte.gz
# Extracting /tmp/data/tlOk-labels-idxl-ubyte.gz
# After 1 training step(s),
# test accuracy = 0.1282 After 1001 training step(s),
# validation accuracy = After 1001 training step(s),
# validation accuracy = After 2001 training step(s),
# validation accuracy = After 3001 training step(s),
# validation accuracy After 4001 training step(s),
# validation accuracy = After 5001 training step(s),
# validation accuracy = After 6001 training step(s),
# validation accuracy = After 6001 training step(s),
# validation accuracy = 0.9769 0.9769 0.9804 0.982 0.983 0.9829 0.9832 0.9832