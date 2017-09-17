import tensorflow as tf
import KaggleLearn.Tools.tools as tools
import numpy as np


# %%
def VGG16(x, n_classes, is_pretrain=True):
    x = tools.conv('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = tools.conv('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = tools.conv('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = tools.conv('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.pool('pool4', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = tools.conv('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.pool('pool5', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = tools.FC_layer('fc6', x, out_nodes=4096)
    x = tf.nn.relu(x)
    # x = tools.batch_norm(x)
    x = tools.FC_layer('fc7', x, out_nodes=4096)
    x = tf.nn.relu(x)
    # x = tools.batch_norm(x)
    x = tools.FC_layer('fc8', x, out_nodes=n_classes)
    x = tf.nn.softmax(x, name="prob")
    return x


# 获取更好的图形效果!
def VGG16N(x, n_classes, is_pretrain=True):
    with tf.name_scope('VGG16'):
        x = tools.conv('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool1'):
            x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool2'):
            x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool3'):
            x = tools.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool4'):
            x = tools.pool('pool4', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool5'):
            x = tools.pool('pool5', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.FC_layer('fc6', x, out_nodes=4096)
        with tf.name_scope('batch_norm1'):
            x = tf.nn.relu(x)
            # x = tools.batch_norm(x)
        x = tools.FC_layer('fc7', x, out_nodes=4096)
        with tf.name_scope('batch_norm2'):
            x = tf.nn.relu(x)
            # x = tools.batch_norm(x)
        x = tools.FC_layer('fc8', x, out_nodes=n_classes)
        x = tf.nn.softmax(x, name="prob")
        return x


def convlayers(x):
    # zero-mean input
    with tf.variable_scope('preprocess') as scope:
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        images = x - mean

    # conv1_1
    with tf.variable_scope('conv1_1') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope.name)

    # conv1_2
    with tf.variable_scope('conv1_2') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool1')

    # conv2_1
    with tf.variable_scope('conv2_1') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope.name)

    # conv2_2
    with tf.variable_scope('conv2_2') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope.name)

    # pool2
    pool2 = tf.nn.max_pool(conv2_2,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool2')

    # conv3_1
    with tf.variable_scope('conv3_1') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope.name)

    # conv3_2
    with tf.variable_scope('conv3_2') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope.name)

    # conv3_3
    with tf.variable_scope('conv3_3') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope.name)

    # pool3
    pool3 = tf.nn.max_pool(conv3_3,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool3')

    # conv4_1
    with tf.variable_scope('conv4_1') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope.name)

    # conv4_2
    with tf.variable_scope('conv4_2') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope.name)

    # conv4_3
    with tf.variable_scope('conv4_3') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope.name)

    # pool4
    pool4 = tf.nn.max_pool(conv4_3,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool4')

    # conv5_1
    with tf.variable_scope('conv5_1') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope.name)

    # conv5_2
    with tf.variable_scope('conv5_2') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope.name)

    # conv5_3
    with tf.variable_scope('conv5_3') as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope.name)

    # pool5
    pool5 = tf.nn.max_pool(conv5_3,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool5')

    with tf.variable_scope('fc6'):
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc1w = tf.get_variable(initializer=tf.truncated_normal([shape, 4096],
                                                               dtype=tf.float32,
                                                               stddev=1e-1), name='weights')
        fc1b = tf.get_variable(initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)

    with tf.variable_scope('fc7'):
        fc2w = tf.get_variable(initializer=tf.truncated_normal([4096, 4096],
                                                               dtype=tf.float32,
                                                               stddev=1e-1), name='weights')
        fc2b = tf.get_variable(initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)

    with tf.variable_scope('fc8'):
        fc3w = tf.get_variable(initializer=tf.truncated_normal([4096, 1000],
                                                               dtype=tf.float32,
                                                               stddev=1e-1), name='weights')
        fc3b = tf.get_variable(initializer=tf.constant(1.0, shape=[1000], dtype=tf.float32),
                               trainable=True, name='biases')
        fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
        fc3l = tf.nn.softmax(fc3l)
    return fc3l
