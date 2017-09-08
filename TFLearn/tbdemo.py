from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf


# 创建权重
def weight_varible(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


# 创建偏好
def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


# 设置卷积
def conv2d(x, W, step=1, name=None):
    return tf.nn.conv2d(x, W, strides=[1, step, step, 1], padding='SAME', name=name)


# 设置池化
def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            # 用直方图记录参数的分布
            tf.summary.histogram('histogram', var)


def add_ConvLayer(inputs, Weights, biases, n_layer, step=1, activation_function=tf.nn.relu):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        # with tf.name_scope('weights'):
            # Weights = weight_varible([patch, patch, in_deep, out_deep], name='W')
            # variable_summaries(Weights)
            # tf.summary.histogram(layer_name + '/weights', Weights)
        # with tf.name_scope('biases'):
            # biases = tf.Variable(tf.zeros([1, depth]) + 0.1, name='b')
            # biases = bias_variable([out_deep], name='b')
            # variable_summaries(biases)
            # tf.summary.histogram(layer_name + '/biases', biases)
        if step == 2:
            conv = conv2d(inputs, Weights, step, layer_name+"Conv")
            linear = tf.add(conv, biases)
            # tf.summary.histogram('linear', linear)
        elif step == 1:
            conv = conv2d(inputs, Weights, step, layer_name+"Conv")
            maxPool = max_pool_2x2(conv, layer_name+"MaxPool")
            linear = tf.add(maxPool, biases)
            # tf.summary.histogram('linear', linear)

        # 激活函数
        activations = activation_function(linear)
        # tf.summary.histogram('activations', activations)
        # tf.summary.histogram(layer_name + '/outputs', outputs)
    return activations


def add_FullLayer(inputs, Weights, biases, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        # with tf.name_scope('weights'):
            # tf.summary.histogram(layer_name + '/weights', Weights)
        # with tf.name_scope('biases'):
            # biases = tf.Variable(tf.zeros([1, depth]) + 0.1, name='b')
            # biases = bias_variable([out_size], name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases, layer_name+"Full")

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        # tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# 卷积模型
def model_conv(data):
    l1 = add_ConvLayer(data, layer1_weights, layer1_biases, step=2, n_layer=1)
    l2 = add_ConvLayer(l1, layer2_weights, layer2_biases, step=2, n_layer=2)
    l3 = add_ConvLayer(l2, layer21_weights, layer21_biases, step=2, n_layer=3)
    shape = l3.get_shape().as_list()
    reshape = tf.reshape(l3, [shape[0], shape[1] * shape[2] * shape[3]])
    l4 = add_FullLayer(reshape, layer3_weights, layer3_biases, n_layer=4)
    l5 = add_FullLayer(l4, layer4_weights, layer4_biases, n_layer=5)

    return l5


# maxpool模型
# 1.1增加深度到32和每一层卷积都增加池化后，卷积步进1，池化步进2，提高到0.98229
# num_hidden从128增加到1024
def model_maxpool(data):
    l1 = add_ConvLayer(data, layer1_weights, layer1_biases, n_layer=1)
    l2 = add_ConvLayer(l1, layer2_weights, layer2_biases, step=1, n_layer=2)
    shape = l2.get_shape().as_list()
    reshape = tf.reshape(l2, [shape[0], shape[1] * shape[2] * shape[3]])
    print("reshape", reshape)
    l3 = add_FullLayer(reshape, layer3_weights, layer3_biases, n_layer=3)
    print("l3", l3)
    # DropOut
    # with tf.name_scope('dropout'):
    keep_prob = 0.7
    # tf.summary.scalar('dropout_keep_probability', keep_prob)
    l3 = tf.nn.dropout(l3, keep_prob)
    l4 = add_FullLayer(l3, layer4_weights, layer4_biases, n_layer=4)
    return l4



batch_size = 32
patch_size = 5
depth = 32
num_hidden = 1024

graph = tf.Graph()
regularation_param = 0.0001

x = tf.placeholder(tf.float32, [None, 784], "x")
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10], "y_")
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

# Tensor board graph (was not in tutorial)
writer = tf.summary.FileWriter("log", sess.graph)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

training_summary = tf.summary.scalar("training_accuracy", accuracy)
validation_summary = tf.summary.scalar("validation_accuracy", accuracy)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 20 == 0:
        # Tensor board graph (was not in tutorial)
        train_acc, train_summ = sess.run(
            [accuracy, training_summary],
            feed_dict={x: mnist.train.images, y_: mnist.train.labels})
        writer.add_summary(train_summ, i)

        valid_acc, valid_summ = sess.run(
            [accuracy, validation_summary],
            feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
        writer.add_summary(valid_summ, i)
