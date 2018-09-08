import tensorflow as tf

# tf.variable_scope 函数生成的上下文管理器也会创建一个 TensorFlow 中的命名空间，在 命名空间内创建的变量名称都会带上这个命名空间名作为前缀。所以，tf.variablescope 函 数除了可以控制 tf.get_variable 执行的功能之外，这个函数也提供了一个管理变量命名空间的方式。以下代码显示了如何通过 tf.variable_scope 来管理变量的名称。
v1 = tf.get_variable("v", [1])
print(v1.name)  # 输出为变量的名称，“:0”表示这个变量是生成变量这个运算的第一个结果。

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", [1])
    print(v2.name)  # 输出foo/v:0。在tf.variable_scope中创建的变量，名称前面会加入命名空间的名称，并通过/来分隔命名空间的名称和变量的名称。

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name)  # 输出foo/bar/v:0。命名空间可以嵌套，同时变量的名称也会加入所有命名空间的名称作为前缀。

    v4 = tf.get_variable("v1", [1])
    print(v4.name)  # 输出 foo/v1:0。当命名空间退出之后，变量名称也就不会再被加入其前缀了。

# 创建一个名称为空的命名空间，并设置 reuse=True
with tf.variable_scope("", reuse=True):
    # 可以直接通过带命名空间名称的变量名来获取其他命名空间下的变量。
    # 比如这里通过指定名称foo/bar/v来获取在命名空间foo/bar/中创建的变量
    v5 = tf.get_variable("foo/bar/v", [1])
    print(v5 == v3)  # 输出True。
    v6 = tf.get_variable("foo/v1", [1])
    print(v6 == v4)  # 输出 True。
