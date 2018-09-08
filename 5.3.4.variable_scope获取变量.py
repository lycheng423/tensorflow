import tensorflow as tf

# tf.get_variable 函数与 tf.Variable 函数最大的区别在于指定变量名称的参数。对于 tf.Variable 函数，变量名称是一个可选的参数，通过 name="v"的形式给出。但是对于tf.get_variable 函数，变量名称是一个必填的参数。tf.get_variable 会根据这个名字去创建或 者获取变量。在上面的样例程序中，tf.get_variable 首先会试图去创建一个名字为 v 的参数, 如果创建失败(比如己经有同名的参数)，那么这个程序就会报错。如果需要通过 tf.get_variable获取一个己经创建的变量，需要通过tf.variable_scope函数来生成一个上下文 管理器，并明确指定在这个上下文管理器中，tf.get_variable 将直接获取己经生成的变量。 下面给出了一段代码说明如何通过 tf.variable_scope 函数来控制 tf.get_variable 函数获取己经创建过的变量。

# 在名字为foo的命名空间内创建名字为V的变量。
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
# 因为在命名空间 foo 中已经存在名字为 v 的变量，所有下面的代码将会报错:
# Variable foo/v already exists, disallowed. Did you mean to set reuse=True in VarScope?
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])

# 在生成上下文管理器时，将参数 reuse 设置为 True。这样 tf.get_variable 函数将直接获取已经声明的变置。
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v == v1)
    # 输出为True，代表v，v1代表的是相同的TensorFlow中变S

    # 将参数reuse设置为True时，tf.variable_scope将只能获取己经创建过的变量。
    # 因为在命名空间bar中还没有创建变量v, 所以下面的代码将会报错:
    # Variable bar/v does not exist, disallowed. Did you mean to set reuse=None in VarScope? _
    with tf.variable_scope("bar", reuse=True):
        v = tf.get_variable("v", [1])
