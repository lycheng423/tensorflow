import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

# 使用 tf.train.Saver 会保存运行 TensorFlow 程序所需要的全部信息，然而有时并不需要 某些信息。
# 比如在测试或者离线预测时，只需要知道如何从神经网络的输入层经过前向传播计算得到输出层即可，
# 而不需要类似于变量初始化、模型保存等辅助节点的信息。在第 6 章介绍迁移学习时，会遇到类似的情况。
# 而且，将变量取值和计算图结构分成不同的文件存储有时候也不方便，
# 于是 TensorFlow 提供了 convert_variables_to_constants 函数，
# 通过这个函数可以将计算图中的变量及其取值通过常量的方式保存，这样整个 TensorFlow 计算 图可以统一存放在一个文件中。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程。
    graph_def = tf.get_default_graph().as_graph_def()

    # 将图中的变量及其取值转化为常量，同时将图中不必要的节点去掉
    # 在5.4.2小节中将会看到一些系统运算也会被转化为计算图中的节点(比如变量初始化操作)。
    # 如果只关心程序中定义的某些计算时，和这些计算无关的节点就没有必要导出并保存了。
    # 在下面一行代码中，最后一个参数['add']给出了需要保存的节点名称。
    # add节点是上面定义的两个变量相加的操作。
    # 注意这里给出的是计算节点的名称，所以没有后面的:0
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

    # 将导出的模型存入文件
    with tf.gfile.GFile("/data/tf/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())


# 通过下面的程序可以直接计算定义的加法运算的结果。
# 当只需要得到计算图中某个节点的取值时，这提供了一个更加方便的方法。
# 第 6 章将使用这种方法来使用训练好的模型完成迁移学习。
with tf.Session() as sess:
    model_filename = "/data/tf/combined_model.pb"
    # 读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer。
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def中保存的图加载到当前的图中。
    # return_elements=["add:0"] 给出了返回的张量的名称。
    # 在保存的时候给出的是计算节点的名称，所以为"add"。
    # 在加载的时候给出的是张量的名称，所以是add:0
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
    # 输出[3.0]