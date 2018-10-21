# saver def属性中记录了持久化模型时需要用到的一些参数，比如保存到文件的文件名、 保存操作和加载操作的名称以及保存频率、清理历史记录等。saver def 属性的类型为 SaverDef, 其定义如下。
message SaverDef {
    string filename_tensor_name 1;
    string save_tensor_name = 2;
    string restore_op_name 3;
    int32 max_to_keep = 4;
    bool sharded = 5;
    float keep_checkpoint_every_n_hours = 6;

    enum CheckpointFormatVersion {
        LEGACY = 0;
        v1 = 1;
        V2 = 2;
    }
    CheckpointFormatVersion version = 7;
}

# 下面给出了model.ckpt.meta.json文件中saver_def属性的内容。
saver_def {
    filename_tensor_name: "save/Const:0"
    save_tensor_name: "save/control_dependency:0"
    restore_op_name: "save/restore all"
    max_to_keep: 5
    keep_checkpoint_every_n_hours: 10000.0
    version: V2
}
# filename_tensor_name 属性给出了保存文件名的张量名称，这个张量就是节点 save/Const的第一个输出。
# save_tensor_name属性给出了持久化TensorFlow模型的运算所对应的节点名称。
# 从上面的文件中可以看出，这个节点就是在 graph_def 属性中给出的 save/control_dependency 节点。
# 和持久化 TensorFlow 模型运算对应的是加载 TensorFlow 模型的运算，这个运算的名称由 restore_op_name 属性指定。
# max_to_keep 属性和 keep_checkpoint_every_n_hours 属性设定了 tf.train.Saver 类清理之前保存的模型的策略。
# 比如当 max_to_keep 为 5 的时候，在第六次调用 saver.save 时，第一次保存的模型就会被自动删除。
# 通过设置 keep_checkpoint_every_n_hours，每n小时可以在 max_to_keep 的基础上多保存一个模型。