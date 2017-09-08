import numpy as np
import pandas as pd
import tensorflow as tf
import time
from sklearn import model_selection

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

# read training data from CSV file
data = pd.read_csv('train.csv')

# print('data({0[0]},{0[1]})'.format(data.shape))
# print(data.head())

images = data.iloc[:, 1:].values
images = images.astype(np.float)

# 数据乘以1/255，0-255->0-1
# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

# print('images({0[0]},{0[1]})'.format(images.shape))

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
# print("train_labels.shape", train_labels.shape)
# print('train_images({0[0]},{0[1]})'.format(train_images.shape))
#
# print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))
# print('validation_labels({0[0]},{0[1]})'.format(validation_images.shape))

train_images, validation_images = model_selection.train_test_split(images, test_size=0.3, random_state=0)
train_labels, validation_labels = model_selection.train_test_split(labels, test_size=0.3, random_state=0)
# print("train_labels.shape", train_labels.shape)
# print('train_images({0[0]},{0[1]})'.format(train_images.shape))
#
# print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))
# print('validation_labels({0[0]},{0[1]})'.format(validation_images.shape))

# read test data from CSV file
test_images = pd.read_csv('test.csv').values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

# print('test_images({0[0]},{0[1]})'.format(test_images.shape))


# 计算精准度
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


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
batch_size = 32
patch_size = 5
depth = 32
num_hidden = 1024
graph = tf.Graph()
regularation_param = 0.0001

# print('train set', train_images.shape, train_labels.shape)
# print('validation set', validation_images.shape, validation_labels.shape)
# print('test set', test_images.shape)
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
# print('train set', train_dataset.shape, train_labels.shape)
# print('validation set', valid_dataset.shape, valid_labels.shape)
# print('test set', test_dataset.shape)


# 创建权重
def weight_varible(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


# 创建偏好
def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


# 设置卷积
def conv2d(x, W, step=1):
    return tf.nn.conv2d(x, W, strides=[1, step, step, 1], padding='SAME')


# 设置池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def add_ConvLayer(inputs, patch, in_deep, out_deep, n_layer, step=1, activation_function=tf.nn.relu):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = weight_varible([patch, patch, in_deep, out_deep], name='W')
            # variable_summaries(Weights)
            # tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            # biases = tf.Variable(tf.zeros([1, depth]) + 0.1, name='b')
            biases = bias_variable([out_deep], name='b')
            # variable_summaries(biases)
            # tf.summary.histogram(layer_name + '/biases', biases)
        if step == 2:
            conv = conv2d(inputs, Weights, step)
            linear = tf.add(conv, biases)
            # tf.summary.histogram('linear', linear)
        elif step == 1:
            conv = conv2d(inputs, Weights, step)
            maxPool = max_pool_2x2(conv)
            linear = tf.add(maxPool, biases)
            # tf.summary.histogram('linear', linear)

        # 激活函数
        activations = activation_function(linear)
        # tf.summary.histogram('activations', activations)
        # tf.summary.histogram(layer_name + '/outputs', outputs)
    return activations


def add_FullLayer(inputs, in_size, out_size, n_layer, activation_function=None):
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
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)

    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)

    conv = tf.nn.conv2d(hidden, layer21_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer21_biases)
    shape = hidden.get_shape().as_list()

    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    hidden = tf.matmul(reshape, layer3_weights)
    hidden = tf.nn.relu(hidden + layer3_biases)

    # l4 = add_FullLayer(reshape, [shape[1] * shape[2] * shape[3], num_hidden], num_hidden, n_layer=4)
    # Weights = weight_varible([shape[1] * shape[2] * shape[3], num_hidden], name='W')
    # print("Weights", Weights)
    # print("layer3_weights", layer3_weights)

    l5 = add_FullLayer(hidden, num_hidden, num_labels, n_layer=5)
    return l5


def model_maxpool(data):
    l1 = add_ConvLayer(data, patch_size, num_channels, depth, n_layer=1)
    l2 = add_ConvLayer(l1, patch_size, depth, depth, step=1, n_layer=2)
    shape = l2.get_shape().as_list()
    reshape = tf.reshape(l2, [shape[0], shape[1] * shape[2] * shape[3]])
    l3 = add_FullLayer(reshape, shape[1] * shape[2] * shape[3], num_hidden, n_layer=3)
    # DropOut
    # with tf.name_scope('dropout'):
    keep_prob = 0.7
    # tf.summary.scalar('dropout_keep_probability', keep_prob)
    l3 = tf.nn.dropout(l3, keep_prob)
    l4 = add_FullLayer(l3, num_hidden, num_labels, n_layer=4)
    return l4

#
# def variable_summaries(var):
#     """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#     with tf.name_scope('summaries'):
#         # 计算参数的均值，并使用tf.summary.scaler记录
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#
#         # 计算参数的标准差
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#             # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
#             tf.summary.scalar('stddev', stddev)
#             tf.summary.scalar('max', tf.reduce_max(var))
#             tf.summary.scalar('min', tf.reduce_min(var))
#             # 用直方图记录参数的分布
#             tf.summary.histogram('histogram', var)


with graph.as_default():
    # 输入数据
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
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

    layer1_weights = weight_varible([patch_size, patch_size, num_channels, depth])
    layer21_weights = weight_varible([patch_size, patch_size, depth, depth])
    layer2_weights = weight_varible([patch_size, patch_size, depth, depth])
    # 全连接层
    layer3_weights = weight_varible([512, num_hidden])
    layer4_weights = weight_varible([num_hidden, num_labels])

    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer21_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # 训练计算
    # 损失函数（loss function）是用来估量你模型的预测值f(x)与真实值Y的不一致程度
    # 它是一个非负实值函数，通常使用L(Y, f(x))来表示，损失函数越小，模型的可能指就越好。
    logits = model_conv(tf_train_dataset)

    #     print(logits.get_shape())# (16, 10)
    #     print(tf_train_labels.get_shape()) # (16, 10)
    with tf.name_scope('loss'):
        hpl2 = regularation_param * (tf.nn.l2_loss(layer1_weights)
                                     + tf.nn.l2_loss(layer21_weights)
                                     + tf.nn.l2_loss(layer2_weights)
                                     + tf.nn.l2_loss(layer3_weights)
                                     + tf.nn.l2_loss(layer4_weights))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        loss = tf.add(loss, hpl2)
        tf.summary.scalar('loss', loss)
        # 计算参数的标准差
    # 计算预测值prediction和真实值的误差
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

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
    optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(loss, global_step=global_step)

    # optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    # 对训练，验证和测试数据集进行预测

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model_conv(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model_conv(tf_test_dataset))

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(tf_train_labels, 1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('accuracy', accuracy1)

with tf.Session(graph=graph) as session:
    num_steps = 301
    EVAL_FREQUENCY = 100
    train_size = train_labels.shape[0]
    start_time = time.time()

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('logs', session.graph)
    tf.global_variables_initializer().run()
    print('Initialized')
    start = time.time()
    for s in range(num_steps):
        offset = (s * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        # _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        summary, l, predictions = session.run([merged, loss, train_prediction], feed_dict=feed_dict)

        if s % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()

            print('Step %d (epoch %.2f), %.1f ms' %
                  (s, float(s) * batch_size / train_size,
                   1000 * elapsed_time / EVAL_FREQUENCY))
            print('Minibatch loss at step %d: %f' % (s, l))

            ma = accuracy(predictions, batch_labels)
            va = accuracy(valid_prediction.eval(), valid_labels)
            print('Minibatch accuracy: %.1f%%' % ma)
            print('Validation accuracy: %.1f%%' % va)
            # with tf.name_scope('accuracy1'):
            #     with tf.name_scope('correct_prediction'):
            #         correct_prediction = accuracy(predictions, batch_labels)
            #     with tf.name_scope('accuracy1'):
            #         accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #     tf.summary.scalar('accuracy1', accuracy1)
                # tf.summary.scalar('Minibatch Accuracy', ma)
                # tf.summary.scalar('Validation Accuracy', va)
            summary_writer.add_summary(summary, s)

    # 获取结果，用于保存
    test_prediction_np = test_prediction.eval()
    test_prediction_np = np.argmax(test_prediction_np, 1)
    savecsv(test_prediction_np, "submission_uda_conv.csv")
    end = time.time()
    print(end - start)
