import tensorflow as tf

# 当 tf.variable_scope 函数使用参数 reuse=True 生成上下文管理器时，
# 这个上下文管理器内所有的tf.get_variable函数会直接获取己经创建的变量。
# 如果变量不存在，则tf.get_variable 函数将报错
# 相反，如果 tf.variable_scope 函数使用参数 reuse=None 或者 reuse=False创建上下文管理器，
# tf.get_variable 操作将创建新的变量。
# 如果同名的变量己经存在，则tf.get_variable函数将报错
# TensorFlow中tf.variable_scope函数是可以嵌套的。
# 下面的程序说明了当tf.variable_scope函数嵌套时，reuse参数的取值是如何确定的。

with tf.variable_scope("root"):
    # 可以通过tf.get_variable_scope().reuse函数来获取当前上下文管理器中reuse参数的取值
    print(1, tf.get_variable_scope().reuse)  # 输出False，即最外层reuse是False。
    # 新建一个嵌套的上下文管理器，并指定reuse为True。
    with tf.variable_scope("foo", reuse=True):
        print(2, tf.get_variable_scope().reuse)  # 输出True。
        # 新建一个嵌套的上下文管理器但不指定reuse，这时reuse的取值会和外面一层保持一致。
        with tf.variable_scope("bar"):
            print(3, tf.get_variable_scope().reuse)  # 输出True。

    print(4, tf.get_variable_scope().reuse)
    # 输出False。退出 reuse 设置为True的上下文之后reuse的值又回到了False。
