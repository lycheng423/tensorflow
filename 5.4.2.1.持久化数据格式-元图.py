import tensorflow as tf

# 5.4.1 小节介绍了当调用 saver.save 函数时，TensorFlow 程序会自动生成 3 个文件。
# TensorFlow 模型的持久化就是通过这 3 个文件完成的。这一小节将详细介绍这 3 个文件中保存的内容以及数据格式。
# 在具体介绍每一个文件之前，先简单回顾一下第 3 章中介绍过 的 TensorFlow 的一些基本概念。
# TensorFlow 是一个通过图的形式来表述计算的编程系统， TensorFlow 程序中的所有计算都会被表达为计算图上的节点
# TensorFlow 通过元图 ( MetaGraph ) 来记录计算图中节点的信息以及运行计算图中节点所需要的元数据。
# TensorFlow中元图是由MetaGraphDef Protocol Buffer定义的。
# MetaGraphDef中的内容就构成了 TensorFlow 持久化时的第一个文件。
# 以下代码给出了 MetaGraphDef 类型的定义。
# message MetaGraphDef {
#    MetalnfoDef meta_info_def = 1;

#    GraphDef graph_def = 2;
#    SaverDef saver_def = 3;

#    map<string, CollectionDef> collection_def = 4;
#    map<string, SignatureDef> signature_def = 5;
#    repeated AssetFileDef asset_file_def = 6;
# }

# 从上面的代码中可以看到，元图中主要记录了 6 类信息。
# 下面的篇幅将结合 5.4.1 小节中变量相加样例的持久化结果，逐一介绍 MetaGmphDef 类型的每一个属性中存储的信息。
# 保存 MetaGraphDef 信息的文件默认以.meta为后缀名，
# 在 5.4.1 小节的样例中，文件model.ckpt.meta 中存储的就是元图的数据。
# 直接运行 5.4.1 小节样例得到的是一个二进制文件，无法直接查看。
# 为了方便调试，TensorFlow 提供了 export_meta_graph 函数，
# 这个函数支持以 json 格式导出 MetaGraphDef Protocol Buffer。
# 以下代码展示了如何使用这个函数。

# 定义变量相加的计算。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
resultl = v1 + v2

saver = tf.train.Saver()
# 通过 export_meta_graph 函数导出 TensorFlow 计算图的元图，并保存为 json 格式。
saver.export_meta_graph("/data/tf/model.ckpt.meda.json", as_text=True)

# 通过上面给出的代码，可以将 5.4.1 小节中的计算图元图以 json 的格式导出并存储在 model.ckpt.meta.json 文件中。下文将结合 model.ckpt.raeta.json 文件具体介绍 TensorFlow 元图中存储的信息。