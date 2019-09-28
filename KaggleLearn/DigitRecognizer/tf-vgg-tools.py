import numpy as np
import pandas as pd
import tensorflow as tf
import time
from sklearn import model_selection

import KaggleLearn.Titanic.testCnn as testCnn
import KaggleLearn.Titanic.VGG as VGG
import KaggleLearn.Tools.tools as tools
import os

# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 25

DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# image number to output
IMAGE_TO_DISPLAY = 10
LOGDIR = "logs"
# read training data from CSV file
data = pd.read_csv('train.csv')

# print('data({0[0]},{0[1]})'.format(data.shape))
# print(data.head())

images = data.iloc[:, 1:].values
images = images.astype(np.float)

# 数据乘以1/255，0-255->0-1
# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))

# print(data)

# labels_flat = data[[0]].values.ravel()
# iiitttttrr
labels_flat = data.iloc[:, 0].values
# 打印标签列长度
# print('labels_flat({0})'.format(len(labels_flat)))
# 打印第X行的图片对应的标签
# print('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY, labels_flat[IMAGE_TO_DISPLAY]))

# unique是返回数组的唯一元素，然后获取形状位置0
labels_count = np.unique(labels_flat).shape[0]
# 打印总数
# print('labels_count = {0}'.format(labels_count))


# 转换类标签标量到独热编码
# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def dense_to_one_hot(labels_dense, num_classes):
    #    labels_dense = [1 0 1 ..., 7 6 9]
    #    num_classes = 10
    num_labels = labels_dense.shape[0]
    #    num_labels = 42000
    index_offset = np.arange(num_labels) * num_classes
    #    index_offset = [     0     10     20 ..., 419970 419980 419990]
    #    index_offset.shape = (42000,)
    #    创建一个全0数组
    labels_one_hot = np.zeros((num_labels, num_classes))
    #     print(labels_one_hot)
    #    labels_one_hot.shape = (42000,10)
    #    flat是一个循环迭代器,通过乘以10，来给每一行的数字打上one hot的1
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    #     print(labels_one_hot.flat[22])
    return labels_one_hot


# print(labels_one_hot.flat[21])
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

# print('labels({0[0]},{0[1]})'.format(labels.shape))
# print('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY, labels[IMAGE_TO_DISPLAY]))

# 分割数据，分成验证集和训练集
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
print("train_labels.shape", train_labels.shape)
print('train_images({0[0]},{0[1]})'.format(train_images.shape))

print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))
print('validation_labels({0[0]},{0[1]})'.format(validation_images.shape))

train_images, validation_images = model_selection.train_test_split(images, test_size=0.3, random_state=0)
train_labels, validation_labels = model_selection.train_test_split(labels, test_size=0.3, random_state=0)
print("train_labels.shape", train_labels.shape)
print('train_images({0[0]},{0[1]})'.format(train_images.shape))
#
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))
print('validation_labels({0[0]},{0[1]})'.format(validation_images.shape))

# read test data from CSV file
test_images = pd.read_csv('test.csv').values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('test_images({0[0]},{0[1]})'.format(test_images.shape))


# 计算精准度
# def accuracy(predictions, labels):
#     return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
#             / predictions.shape[0])

def accuracy1(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor,
  """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + '/accuracy', accuracy)
    return accuracy


# 保存结果
def savecsv(test_prediction_np, savename='submission_uda_dr.csv'):
    np.savetxt(savename,
               np.c_[range(1, len(test_images) + 1), test_prediction_np],
               delimiter=',',
               header='ImageId,Label',
               comments='',
               fmt='%d')


image_size = 28
num_labels = 10
num_channels = 1  # grayscale

print('train set', train_images.shape, train_labels.shape)
print('validation set', validation_images.shape, validation_labels.shape)
print('test set', test_images.shape)
valid_labels = validation_labels


def reformat2(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def reformat1(dataset):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset


train_dataset = reformat1(train_images)
valid_dataset = reformat1(validation_images)
test_dataset = reformat1(test_images)
print('train set', train_dataset.shape, train_labels.shape)
print('validation set', valid_dataset.shape, valid_labels.shape)
print('test set', test_dataset.shape)


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


def add_FullLayer1(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = weight_varible([in_size, out_size], name='W')
            # tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            # biases = tf.Variable(tf.zeros([1, depth]) + 0.1, name='b')
            biases = bias_variable([out_size], name='b')
            # tf.summary.histogram(layer_name + '/biases', biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

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
# maxpool 1.2，1.2 + 1024   32，5，32，1024 =  0.97914
# 加入L2  0.98143


with graph.as_default():
    # 输入数据
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels), name="x")
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name="labels")

    tf_valid_dataset = tf.constant(valid_dataset)
    # tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels), name="v")
    # tf_valid_labels = tf.constant(valid_labels)
    tf_valid_labels = tf.placeholder(tf.float32, shape=valid_labels.shape, name="vlabels")

    tf_test_dataset = tf.constant(test_dataset)

    # 转数据类型，f64->f32
    tf_train_dataset = tf.to_float(tf_train_dataset)
    tf_valid_dataset = tf.to_float(tf_valid_dataset)
    tf_test_dataset = tf.to_float(tf_test_dataset)

    # 变量，在这里是过滤器用
    # truncated_normal按照正态分布初始化权重
    # mean是正态分布的平均值
    # stddev是正态分布的标准差（standard deviation）
    # seed是作为分布的random seed（随机种子，我百度了一下，跟什么伪随机数发生器还有关，就是产生随机数的）

    layer1_weights = weight_varible([patch_size, patch_size, num_channels, depth], name='W')
    layer21_weights = weight_varible([patch_size, patch_size, depth, depth], name='W')
    layer2_weights = weight_varible([patch_size, patch_size, depth, depth], name='W')
    layer3_weights = weight_varible([1568, num_hidden], name='W')
    layer4_weights = weight_varible([num_hidden, num_labels], name='W')

    layer1_biases = bias_variable([depth], name='b')
    layer21_biases = bias_variable([depth], name='b')
    layer2_biases = bias_variable([depth], name='b')
    layer3_biases = bias_variable([num_hidden], name='b')
    layer4_biases = bias_variable([num_labels], name='b')

    # 训练计算
    # 损失函数（loss function）是用来估量你模型的预测值f(x)与真实值Y的不一致程度
    # 它是一个非负实值函数，通常使用L(Y, f(x))来表示，损失函数越小，模型的可能指就越好。
    # fc1 = model_maxpool(tf_train_dataset)[0]

    train_logits = model_maxpool(tf_train_dataset)
    valid_logits = model_maxpool(tf_valid_dataset)
    test_logits = model_maxpool(tf_test_dataset)

    train_prediction = tf.nn.softmax(train_logits, name="train_prediction")
    valid_prediction = tf.nn.softmax(valid_logits, name="valid_prediction")
    test_prediction = tf.nn.softmax(test_logits, name="test_prediction")

    #     print(logits.get_shape())# (16, 10)
    #     print(tf_train_labels.get_shape()) # (16, 10)
    with tf.name_scope('loss'):
        hpl2 = regularation_param * (tf.nn.l2_loss(layer1_weights)
                                     + tf.nn.l2_loss(layer21_weights)
                                     + tf.nn.l2_loss(layer2_weights)
                                     + tf.nn.l2_loss(layer3_weights)
                                     + tf.nn.l2_loss(layer4_weights))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=train_logits))
        loss = tf.add(loss, hpl2)

        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        tf.summary.scalar('loss', loss)

    # 初始的学习速率
    starter_learning_rate = 0.1
    # 全局的step，与 decay_step 和 decay_rate一起决定了 learning rate的变化
    global_step = tf.Variable(0, trainable=False)
    # 衰减速度
    decay_steps = 50
    # 衰减系数
    decay_rate = 0.9
    # 如果staircase=True，那就表明每decay_steps次计算学习速率变化，更新原始学习速率.
    # 如果是False，那就是每一步都更新学习速率
    staircase = False
    # 指数衰减:法通过这个函数，可以先使用较大的学习率来快速得到一个比较优的解，然后随着迭代的继续逐步减小学习率，使得模型在训练后期更加稳定
    # 87.7% 仅仅指数衰减
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase)

    # 优化器
    # optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(loss, global_step=global_step)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    accuracy = accuracy1(train_logits, tf_train_labels)
        # tf.summary.scalar('optimizer', optimizer)
    # 对训练，验证和测试数据集进行预测



num_steps = 50
EVAL_FREQUENCY = 100
train_size = train_labels.shape[0]
start_time = time.time()
# config = tf.ConfigProto(device_count={"CPU": 1},  # limit to num_cpu_core CPU usage
#                         inter_op_parallelism_threads=1,
#                         intra_op_parallelism_threads=2,
#                         log_device_placement=True)

with tf.Session(graph=graph) as session:
    train_writer = tf.summary.FileWriter(LOGDIR + '/train', session.graph)
    valid_writer = tf.summary.FileWriter(LOGDIR + '/valid')

    tools.print_all_variables()

# with tf.Session(config=config, graph=graph) as session:
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(LOGDIR, session.graph)
    # train_writer.add_graph(session.graph)
    tf.global_variables_initializer().run()
    print('Initialized')
    start = time.time()
    for s in range(num_steps):
        if s % 5 == 0:
            offset = (s * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data_train = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels_train = train_labels[offset:(offset + batch_size), :]

            batch_data_valid = valid_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels_valid = valid_labels[offset:(offset + batch_size), :]

            # print(batch_data_valid.shape)
            # print(batch_labels_valid.shape)

            feed_dict_t = {tf_train_dataset: batch_data_train, tf_train_labels: batch_labels_train}
            feed_dict_v = {tf_train_dataset: batch_data_valid, tf_train_labels: batch_labels_valid}
            # _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict_t)
            summary_t, _, l, t_predictions = session.run([merged, optimizer, loss, train_prediction], feed_dict=feed_dict_t)
            summary_v, _, l, v_predictions = session.run([merged, optimizer, loss, valid_prediction], feed_dict=feed_dict_v)

        # if s % 20 == 0:
        #     train_acc, train_summ = session.run([accuracy_train, training_summary], feed_dict=feed_dict_t)
        #     training_summary = tf.summary.scalar("training_accuracy", accuracy)
            train_acc = session.run([accuracy], feed_dict=feed_dict_t)
            train_writer.add_summary(summary_t, s)
            valid_writer.add_summary(summary_v, s)
            # val_images, val_labels = session.run([batch_data_valid, batch_labels_valid])
            # val_loss, val_acc = session.run([loss, accuracy], feed_dict={tf_valid_dataset: val_images, tf_valid_labels: val_labels})
            # print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (s, val_loss, val_acc))

            # summary_str = session.run(merged)
            # valid_writer.add_summary(summary_str, s)

            # valid_acc, valid_summ = session.run([accuracy_valid, validation_summary], feed_dict=feed_dict_v)
            # train_writer.add_summary(valid_summ, s)

        if s % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.1f ms' %
                  (s, float(s) * batch_size / train_size,
                   1000 * elapsed_time / EVAL_FREQUENCY))
            print('Minibatch loss at step %d: %f' % (s, l))
            # valid_acc, valid_summ = session.run([valid_prediction, validation_summary], feed_dict=feed_dict_v)
            # writer.add_summary(valid_summ, step)
            ta = accuracy1(t_predictions, batch_labels_train)
            va = accuracy1(valid_prediction.eval(), valid_labels)



        # if s % 200 == 0:
        #     # session.run(assignment, feed_dict=feed_dict)
        #     saver.save(session, os.path.join(LOGDIR, "model.ckpt"), s)

    # 获取结果，用于保存
    test_prediction_np = test_prediction.eval()
    test_prediction_np = np.argmax(test_prediction_np, 1)
    savecsv(test_prediction_np, "submission_uda_conv.csv")
    end = time.time()
    print(end - start)
