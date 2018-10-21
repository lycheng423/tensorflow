# meta_info_def 属性是通过 MetalnfoDef 定义的，
# 它记录了 TensorFlow 计算图中的元数据以及 TensorFlow 程序中所有使用到的运算方法的信息。
# 下面是 MetalnfoDef Protocol Buffer 的定义:

message MetalnfoDef {
    string meta_graph_version = 1;
    OpList stripped_op_list = 2;
    google.protobuf.Any any_info = 3;
    repeated string tags = 4;
    string tensorflow_version = 5;
    string tensorflow_git_version = 6;
}

# TensorFlow 计算图的元数据包括了计算图的版本号(meta_graph_version 属性)
# 以及用户指定的一些标签(tags 属性)。如果没有在 saver 中特殊指定，那么这些属性都默认为空。
# 在model.ckpt.meta.json文件中，meta_info_def属性里只有stripped_op_list属性是不为空的。
# stripped_op_list 属性记录了 TensorFlow 计算图上使用到的所有运算方法的信息。
# 注意 stripped_op_list 属性保存的是 TensorFlow 运算方法的信息，
# 所以如果某一个运算在 TensorFlow 计算图中出现了多次，那么在 stripped_op_list 也只会出现一次。
# 比如在 model.ckpt.meta.json文件的stripped_op_list属性中只有一个Variable运算，但这个运算在程序中被使用了两次。
# 属性的类型是OpList。OpList类型是一个OpDef类型的列表，以下代码给出了 OpDef 类型的定义:
message OpDef {
    string name = 1;

    repeated ArgDef input_arg = 2;
    repeated ArgDef output_arg = 3;
    repeated AttrDef attr = 4;

    OpDeprecation deprecation = 8;
    string summary = 5;
    string description = 6;
    bool is_commutative = 18;
    bool is_aggregate = 16;
    bool is_stateful = 17;
    bool allows_uninitialized_input = 19;
}
# OpDef 类型中前四个属性定义了一个运算最核心的信息。
# OpDef 中的第一个属性 name 定义了运算的名称，这也是一个运算唯一的标识符。
# 在 TensorFlow 计算图元图的其他属性 中，比如下面将要介绍的 GraphDef 属性，将通过运算名称来引用不同的运算。
# OpDef 的第二和第三个属性为 input_arg 和 output_arg，它们定义了运算的输入和输出。
# 因为输入输出都可以有多个，所以这两个属性都是列表(repeated)。
# 第四个属性 attr 给出了其他的运算参数信息。
# 在 model.ckpt.meta.json 文件中总共定义了 8 个运算，下面将给出比较有代表性 的一个运算来辅助说明 OpDef 的数据结构。
op {
    name: "Add"
    input_arg {
        name: "x"
        type_attr: "T"
    }
    input_arg {
        name: "y"
        type_attr: "T"
    }
    output_arg {
        name: "z"
        type_attr: "T"
    }
    attr {
        name: "T"
        type: "type"
        allowed_values {
            list {
                type: DT_HALF
                type: DT_FLOAT
                ...
            }
        }
    }
}
# 上面给出了名称为 Add 的运算。
# 这个运算有 2 个输入和 1 个输出，输入输出属性都指 定了属性 type_attr,并且这个属性的值为 T。
# 在 OpDef 的 attr 属性中，必须要出现名称( name) 为 T 的属性。
# 以上样例中，这个属性指定了运算输入输出允许的参数类型(allowectvalues)。

# tensorflow_version 和 tensorflow_git_version 记录了生成当前计算图的TF版本