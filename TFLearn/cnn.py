from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 本CNN分四个layer
# 1. convolutional layer1 + max pooling
# 2. convolutional layer2 + max pooling
# 3. fully connected layer + dropout
# 4. fully connected layer:readout layer to prediction

# 为输入值设立placeholder
x = tf.placeholder(tf.float32, [None, 784])  # 28 * 28
y_ = tf.placeholder(tf.float32, [None, 10])
# 将输入x的形状变为[-1, 28, 28, 1], -1表示不考虑输入的图片例子多少这个维度，因为为黑白图像，所以channel为1
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 权重、偏置初始化
# truncated_nomal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# 截尾正态分布，抛弃偏离均值两倍标准差的值
# 权重应该初始很小但不能为0，应该有对称分布的噪声来避免0梯度值的出现
# 偏置初始化为一个略大于0的值，避免在ReLU激活函数下"死神经元"节点在一开始就出现
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积和池化
# （1）关于卷积：
#     输入x的格式：[batch, in_height, in_width, in_channels]
#     卷积核(也叫filter)的格式:[filter_height, filter_width, in_channels, out_channels]
#     在conv2d中，x和W进行右乘
#     步长:strides: strides[0]和strides[3]的两个1是默认值，中间第二个值和第三个值为在水平方向和竖直方向移动的步长
#     padding = 'SAME' 表示输出图像和输入图像等大小
# （2）关于池化：
#     池化有两种，最大值池化和平均值池化，这里采用最大值池化
#     ksize是核，核函数大小为2*2
# 关于卷积的计算：参见https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#convolution
# 其实，tensorflow中的卷积对核并没有翻转
# conv2d用的filter(核)是随机的
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积层
# 卷积核为[5, 5, 1, 32], patch-size:5*5, 输出32个特征值
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 开始卷积和池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output_size: 28*28*32
h_pool1 = max_pool_2x2(h_conv1)  # output_size: 14*14*32

# 第二层卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output_size: 14*14*64
h_pool2 = max_pool_2x2(h_conv2)  # output_size:7*7*64

# 全连接层
# 通过tf.reshape()将h_pool2的输出值从三维数据变为一维的数据h_pool2_flat，进行数据展平，即：
# [batch_size, 7, 7, 64] ->> [batch_size, 7 * 7 * 64]
# -1 表示先不考虑batch size
# 将展平后的h_pool2_flat与本层的W_fc1相乘并加上偏置，注意此时不是卷积了
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 考虑过拟合问题，添加dropout进行处理
# keep_prob为要保留的比例
# 训练时打开dropout, 测试时关闭dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 最后一层：Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 训练并评估模型
with tf.Session() as sess:
    # 定义损失函数和准确度
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))  # 用交叉熵定义损失函数
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 选用AdamOptimizer
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 返回boolean值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())  # 初始化变量

    writer = tf.summary.FileWriter("log", sess.graph)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    training_summary = tf.summary.scalar("training_accuracy", accuracy)
    validation_summary = tf.summary.scalar("validation_accuracy", accuracy)

    for i in range(200):
        batch = mnist.train.next_batch(50)  # 一次取50个batch训练
        print(i)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 20 == 0:
            # Tensor board graph (was not in tutorial)
            train_acc, train_summ = sess.run(
                [accuracy, training_summary],
                feed_dict={x: mnist.train.images, y_: mnist.train.labels})
            writer.add_summary(train_summ, i)

            # valid_acc, valid_summ = sess.run(
            #     [accuracy, validation_summary],
            #     feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
            # writer.add_summary(valid_summ, i)

            # print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
