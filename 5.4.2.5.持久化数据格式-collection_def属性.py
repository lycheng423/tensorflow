#在 TensorFlow 的计算图 ( tf.Graph ) 中可以维护不同集合，而维护这些集合的底层实现就是通过 collection_def 这个属性。
# collection_def 属性是一个从集合名称到集合内容的映射，其中集合名称为字符串，
# 而集合内容为 CollectionDef Protocol Buffer。以下代码给出了 CollectionDef 类型的定义。
message CollectionDef {
    message NodeList {
        repeated string value = 1;
    }
    message BytesList {
        repeated bytes value = 1;
    }
    message Int64List {
        repeated int64 value = 1 [packed = true];
    }
    message FloatList {
        repeated float value = 1 [packed = true];
    }
    message AnyList {
        repeated google.protobuf.Any value = 1;
    }
    oneof kind {
        NodeList node_list = 1;
        BytesList bytes_list = 2;
        Int64List int64_list = 3;
        FloatList float_list = 4;
        AnyList any_list = 5;
    }
}
#通过上面的定义可以看出，TensorFlow 计算图上的集合主要可以维护 4 类不同的集合。
# NodeList 用于维护计算图上节点的集合。BytesList 可以维护字符串或者系列化之后的 Procotol Buffer 的集合。
# 比如张量是通过 Protocol Buffer 表示的，而张量的集合是通过 BytesList 维护的，我们将在 model.ckpt.meta.json 文件中看到具体样例。
# Int64List 用于维护 整数集合，FloatList 用于维护实数集合。下面给出了 model.ckpt.meta.json 文件中 collection_def 属性的内容。

collection_def {
    key: "trainable_variables"
    value {
        bytes_list {
            value: "\n\004v1:0\022\tvl/Assign\032\tv1/read:0"
            value: "\n\004v2:0\022\tv2/Assign\032\tv2/read:0"
        }
    }
    collection_def {
        key: "variables"
        value {
            bytes_list {
                value: "\n\004v1:0\022\tvl/Assign\032\tv1/read:0"
                value: "\n\004v2:0\022\tv2/Assign\032\tv2/read:0"
            }
        }
    }

# 从上面的文件可以看出样例程序中维护了两个集合。一个是所有变量的集合，这个集合的名称为 variables。
# 另外一个是可训练变量的集合，名为trainable_variables。
# 在样例程序中，这两个集合中的元素是一样的，都是变量 v1 和 v2。它们都是系统自动维护的