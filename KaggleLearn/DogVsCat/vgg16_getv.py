########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from KaggleLearn.DogVsCat.imagenet_classes import class_names


class vgg16_getv:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        # self.convlayers()
        # self.fc_layers()
        f3 = vgg16_getv.convlayers()
        self.probs = tf.nn.softmax(f3)
        if weights is not None and sess is not None:
            vgg16_getv.load_weights_tf(weights, sess)

    def convlayers(self):
        # zero-mean input
        with tf.variable_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = imgs - mean

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

        return fc3l

    def load_weights_tf(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print("keys", keys)
        for key in keys:
            x = str(key[:-2])
            with tf.variable_scope(key[:-2], reuse=True):
                if key[-1:] is "W":
                    subkey = "weights"
                else:
                    subkey = "biases"
                xxx = weights[key]
                print("k,np.shape", key, np.shape(weights[key]))
                sess.run(tf.get_variable(subkey).assign(weights[key]))

    def load_weights_npy_tf(self, weight_file, sess):
        weights = np.load(weight_file, encoding='latin1').item()
        keys = sorted(weights.keys())
        print("keys", keys)
        for i, k in enumerate(keys):
            for subkey in (0, 1):
                # xxx1 = weights[k][subkey]
                # xxx2 = weights[k][subkey]
                # print("xxx1.shape", xxx1.shape)
                # print("xxx2.shape", xxx2.shape)
                # print("i,k，subkey,np.shape", j, k, subkey, np.shape(weights[k][subkey]))
                # sess.run(parameters[j].assign(weights[k][subkey]))
                sess.run(tf.get_variable(subkey).assign(weights[subkey]))
                # sess.run(parameters[j].assign(weights[k][subkey]))
                # j += 1


if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16_getv(imgs, './VGG16/vgg16_weights.npz', sess)
    # vgg = vgg16(imgs, './VGG16/vgg16.npy', sess)

    img1 = imread('./test_data/tiger.jpeg', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])
# down load websiet: https://www.cs.toronto.edu/~frossard/
