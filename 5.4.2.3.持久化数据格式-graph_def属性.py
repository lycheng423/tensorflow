# graph_def 属性主要记录了 TensorFlow 计算图上的节点信息。
# TensorFlow 计算图的每一个节点对应了 TensorFlow 程序中的一个运算。
# 因为在 meta_info_def 属性中已经包含了所有运算的具体信息，所以 graph_def 属性只关注运算的连接结构。
# graph_def 属性是通过 GraphDef Protocol Buffer 定义的，GraphDef 主要包含了一个 NodeDef 类型的列表。
# 以下代码给出了 GraphDef 和 NodeDef 类型中包含的信息:
message GraphDef {
    repeated NodeDef node = 1;
    VersionDef versions = 4;
}

message NodeDef {
    string name = 1;
    string op = 2;
    repeated string input = 3;
    string device = 4;
    map<stringf AttrValue> attr = 5;
}
# GraphDef 中的 versions 属性比较简单，它主要存储了 TensorFlow 的版本号。
# GraphDef 的主要信息都存在 node 属性中，它记录了 TensorFlow 计算图上所有的节点信息。
# 和其他属性类似，NodeDef 类型中有一个名称属性 name, 它是一个节点的唯一标识符。
# 在 TensorFlow 程序中可以通过节点的名称来获取相应的节点。
# NodeDef 类型中的 op 属性给出 了该节点使用的 TensorFlow 运算方法的名称，
# 通过这个名称可以在 TensorFlow 计算图元图 的 meta_info_def 属性中找到该运算的具体信息。
# NodeDef 类型中的 input 属性是一个字符串列表，它定义了运算的输入。
# input 属性中每个字符串的取值格式为 node:src_output,
# 其中 node 部分给出了一个节点的名称，src_output 部分表明了这个输入是指定节点的第几个输出。
# 当 src_output 为 0 时，可以省略:src_output 这个部分。
# 比如 node:0 表示名称为 node 的节点的第一个输出，它也可以被记为 node。
# NodeDef 类型中的 device 属性指定了处理这个运算的设备。
# 运行 TensorFlow 运算的设 备可以是本地机器的 CPU 或者 GPU, 也可以是一台远程的机器 CPU 或者 GPU。
# 当 device 属性为空时，TensorFlow 在运行时会自动选取一个最合适的设备来运行这个运算。
# 最后 NodeDef 类型中的 attr 属性指定了和当前运算相关的配置信息。
# 下面列举了 modcl.ckpt.meta.json 文件中的一些计算节点来更加具体地介绍 graph_def 属性。
graph_def {
    node {
        name: "v1"
        op: "VariableV2"
        attr {
            key: "_output_shapes"
            value {
                list{   shape{  dim{    size:1  }   }   }
            }
        }
        attr {
            key: "dtype"
            value {
                type: DT_FLOAT
            }
        }
        ...
    }
    node {
        name: "add"
        op: "Add"
        input: "v1/read"
        input: "v2/read"
        ...

    }
    node {
        name "save/control_dependency"
        op: "Identity"
        ...
    }
    versions {
        producer 24
    }
}

# 上面给出了model.ckpt.meta.json文件中graph def属性里比较有代表性的几个节点。
# 第一个节点给出的是变量定义的运算。
# 在 TensorFlow 中变量定义也是一个运算，这个运算的名称为 v1(name: "v1")
# 运算方法的名称为 Variable (op: "VariableV2”)。
# 定义变量的运算可以有很多个，于是在 NodeDef 类型的 node 属性中可以有多个变量定义的节点。
# 但定义变量的运算方法只用到了一个，于是在 MetalnfoDef 类型的 stripped_op_list 属性中只有一个名称为 Variable 的运算方法。
# 除了指定计算图中节点的名称和运算方法，NodeDef 类型中还定义了运算相关的属性。
# 在节点v1中，attr 属性指定了这个变量的维度以及类型。
# 给出的第二个节点是代表加法运算的节点。它指定了 2 个输入，一个为 v1/read, 另一个为 v2/read。
# 其中 v1/read 代表的节点可以读取变量 v1 的值。
# 因为 v1 的值是节点 v1/read 的第一个输出，所以后面的:0 就可以省略了。
# v2/read 也类似的代表了变量 v2 的取值。
# 以上样例文件中给出的最后一个名称为 save/control_dependency，该节点是系统在完成 TensorFlow 模型持久化过程中自动生成的一个运算。
# 在样例文件的最后，属性 给出了生成 model ckpt.meta.json 文件时使用的 TensorFlow 版本号。