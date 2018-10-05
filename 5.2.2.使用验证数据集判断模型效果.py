# 在 5.2.1 小节给出了使用神经网络解决 MNIST 问题的完整程序。
# 在这个程序的开始设置了初始学习率、学习率衰减率、隐藏层节点数量、迭代轮数等 7 种不同的参数。
# 在大部分情况下, 配置神经网络的这些参数都是需要通过实验来调整的。
# 为了估计模型在未知数据上的效果，需要保证测试数据在训练过程中是不可见的。
# 一般会从训练数据中抽取一部分作为验证数据，使用验证数据就可以评判不同参数取值下模型的表现

# 为了说明验证数据在一定程度上可以作为模型效果的评判标准，
# 我们将对比在不同迭代轮数的情况下，模型在验证数据和测试数据上的正确率。
# 为了同时得到同一个模型在验证数据和测试数据上的正确率，可以在每 1000 轮的输出中加入在测试数据集上的正确率。
# 在 5.2.1 小节给出的代码中加入以下代码，就可以得到每 1000 轮迭代后，
# 使用了滑动平均的模型在验证数据和测试数据上的正确率。

# 计算滑动平均模型在测试数据和验证数据上的正确率。
validate_acc = sess.run(accuracy, feed_dict=validate_feed)
test_acc = sess.run(accuracy, feed_dict=test_feed)

# 输出正确率信息。
print("After %d training step(s), validation accuracy "
      "using average Mmodel is %g test accuracy using average model is %g" %
      (i, validatea_cc, test_acc))
