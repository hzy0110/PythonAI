from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
# 我是在Jupyter Notebook里运行的，所以需要这行
# %matplotlib inline

# 读入数据
train = pd.read_csv("input/train.csv")
# 选取房屋面积小于１２０００的数据
train = train[train['LotArea'] < 12000]
train_X = train['LotArea'].values.reshape(-1, 1)
train_Y = train['SalePrice'].values.reshape(-1, 1)

n_samples = train_X.shape[0]
# 学习率
learning_rate = 2
# 迭代次数
training_epochs = 1000
# 每多少次输出一次迭代结果
display_step = 50

# 这个X和Y和上面的train_X,train_Y是不一样的，这里只是个占位符，
# 训练开始的时候需要“喂”(feed)数据给它
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# 定义模型参数
W = tf.Variable(np.random.randn(), name="weight", dtype=tf.float32)
b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)

# 定义模型
pred = tf.add(tf.multiply(W, X), b)
# 定义损失函数
cost = tf.reduce_sum(tf.pow(pred-Y, 2)) / (2 * n_samples)
# 使用Adam算法，至于为什么不使用一般的梯度下降算法，一会说
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.initialize_all_variables()

# 训练开始
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.3f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # 画图
    plt.plot(train_X, train_Y, 'ro', label="Original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()