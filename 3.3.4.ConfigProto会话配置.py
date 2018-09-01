import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True,  # GPU 可移植到 CPU
                        log_device_placement=True)  # 记录日志，每个节点在那个设备上

sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
