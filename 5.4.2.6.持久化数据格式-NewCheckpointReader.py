import tensorflow as tf

#通过对 MetaGraphDef 类型中主要属性的讲解，本小节己经介绍了 TensorFlow 模型持久化得到的第一个文件中的内容。
# 除了持久化 TensorFlow 计算图的结构,持久化 TensorFlow 中变量的取值也是非常重要的一个部分。
# 5.4.1 小节中使用 tf.Saver 得到的 model.ckpt.index 和 model.ckpt.data-***-of-*** 文件就保存了所有变量的取值。
# 其中 model.ckpt.data 文件是通过 SSTable 格式存储的，可以大致理解为就是一个(key, value) 列表。
# TensroFlow 提供了 tf.train.NewCheckpointReader 类来查看保存的变量信息。
# 以下代码展示了如何使用 tf.train.NewCheckpointReader 类。

# tf.train.NewCheckpointReader可以读取checkpoint文件中保存的所有变量。
# 注意后面的.data 和 .index 可以省去
reader = tf.train.NewCheckpointReader('/data/tf/model.ckpt')
# 获取所有变量列表。这个是一个从变量名到变量维度的字典。
global_variables = reader.get_variable_to_shape_map()
for variable_name in global_variables:
    # variable 为变量名称，global_variables[variable]为变量的维度。
    print(variable_name, global_variables[variable_name])

# 获取名称为 v1 的变量的取值。
print("Value for variable v1 is ", reader.get_tensor("v1"))

#...
#这个程序将输出:
#v1 [1]  变量v1的维度为[1]
#v2 [1]  变量v2的维度为[1]
# Value for variable v1 is [ 1.] 变量 v1 的取值为 1
# ...


# 最后一个文件的名字是固定的，叫 checkpoint。这个文件是 tf.train.Saver 类自动生成且自动维护的。
# 在 checkpoint 文件中维护了由一个 tf.train.Saver 类持久化的所有 TensorFlow模型文件的文件名。
# 当某个保存的 TensorFlow 模型文件被删除时，这个模型所对应的文件名也会从 checkpoint 文件中删除。
# checkpoint 中内容的格式为 CheckpointState Protocol Buffer,
# 下面给出了 CheckpointState类型的定义。

#message CheckpointState {
#    string model_checkpoint_path = 1;
#    repeated string all_model_checkpoint_paths = 2;
#}

# model_checkpoint_path 属性保存了最新的 TensorFlow 模型文件的文件名。
# all_model_checkpoint_paths 属性列出了当前还没有被删除的所有 TensorFlow 模型文件的文件名。
# 下面给出了通过 5.4.1 节中样例程序生成的 checkpoint 文件。
# model_checkpoint_path: "/data/tf/model.ckpt"
# all_model_checkpoint_paths: "/data/tf/model.ckpt"