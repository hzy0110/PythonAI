import tensorflow as tf
import numpy as np


# %%
def conv(layer_name, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=True):
    """Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    """

    in_channels = x.get_shape()[-1]
    # print(type(in_channels))
    with tf.variable_scope(layer_name):
        # w = tf.get_variable(name='weights',
        # trainable=is_pretrain,
        # shape=[kernel_size[0], kernel_size[1], in_channels, out_channels]) # default is uniform distribution initialization
        w = tf.get_variable(name='weights', initializer=tf.truncated_normal([kernel_size[0], kernel_size[1], int(in_channels), out_channels], dtype=tf.float32, stddev=1e-1))
        b = tf.get_variable(initializer=tf.constant(0.0, shape=[out_channels], dtype=tf.float32), trainable=is_pretrain, name='biases')
        # b = tf.get_variable(name='biases',
        #                     trainable=is_pretrain,
        #                     shape=[out_channels])
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


# %%
def pool(layer_name, x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True, padding='SAME'):
    """Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    """
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding=padding, name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding=padding, name=layer_name)
    return x


# %%
def batch_norm(x):
    """Batch normlization(I didn't include the offset and scale)
    """
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


# %%
def FC_layer(layer_name, x, out_nodes):
    """Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    """
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        # w = tf.get_variable('weights', shape=[size, out_nodes])
        w = tf.get_variable(initializer=tf.truncated_normal([size, out_nodes], dtype=tf.float32, stddev=1e-1), name='weights')
        # b = tf.get_variable('biases', shape=[out_nodes])
        b = tf.get_variable(initializer=tf.constant(1.0, shape=[out_nodes], dtype=tf.float32), trainable=True, name='biases')
        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        # x = tf.nn.relu(x)
        return x


# %%
def loss(logits, labels):
    """Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    """
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope + '/loss', loss)
        return loss


# %%
def accuracy(logits, labels):
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


# %%
def num_correct_prediction(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Return:
    the number of correct predictions
    """
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct


# %%
def optimize(loss, learning_rate, global_step):
    """optimization, use Gradient Descent as default
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


# %%
def loadnpz(data_path, session):
    """
    读取保存的参数
    :param data_path:
    :param session:
    :return:
    """
    data_dict = np.load(data_path)

    keys = sorted(data_dict.keys())
    for key in keys:
        # print("key", key)
        # print(type(key))
        with tf.variable_scope(key[:-2], reuse=True):
            # subkey = ""
            # print("key[-1:]", key[-1:])
            if key[-1:] is "W":
                subkey = "weights"
            else:
                subkey = "biases"
            # for subkey, data in zip(('weights', 'biases'), data_dict[key]):
            # for subkey in zip(('weights', 'biases')):
            xx1 = data_dict[key]
            # xx2 = data
            # print("key", key)
            # print("subkey", subkey)
            print("key, subkey, np.shape(data)=", key, subkey,  np.shape(data_dict[key]))
            xx1 = tf.get_variable(subkey)
            session.run(tf.get_variable(subkey).assign(data_dict[key]))
                # xx2 = tf.get_variable(subkey)
                # a = 1
    # for i, k in enumerate(keys):
    #     print("k=", k)
    #     # print("np.shape(data_dict[k])=", np.shape(data_dict[k]))
    #     xxx = data_dict[k]
    #     print("type(data_dict[k])=", type(data_dict[k]))

        # print("i,k,np.shape", i, k, np.shape(data_dict[k]))
        # sess.run(self.parameters[i].assign(data_dict[k]))


def load(data_path, session):
    """
    读取保存的参数
    :param data_path:
    :param session:
    :return:
    """
    data_dict = np.load(data_path, encoding='latin1').item()

    keys = sorted(data_dict.keys())
    for key in keys:
        # print("key", key)
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                xx1 = data_dict[key]
                xx2 = data
                print("subkey, key, np.shape(data)=", subkey, key, np.shape(data))
                # xx1 = tf.get_variable(subkey)
                print("subkey", subkey)
                session.run(tf.get_variable(subkey).assign(data))
                # xx2 = tf.get_variable(subkey)
                # a = 1
                # for i, k in enumerate(keys):
                #     print("k=", k)
                #     # print("np.shape(data_dict[k])=", np.shape(data_dict[k]))
                #     xxx = data_dict[k]
                #     print("type(data_dict[k])=", type(data_dict[k]))

                # print("i,k,np.shape", i, k, np.shape(data_dict[k]))
                # sess.run(self.parameters[i].assign(data_dict[k]))


# %%
def test_load():
    """
    读取并打印VGG16的形状
    :return:
    """
    data_path = './vgg16/vgg16.npy'

    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)


def load_with_skip(data_path, session, skip_layer):
    """
    控制哪几层的参数不加载
    :param data_path:
    :param session:
    :param skip_layer:
    :return:
    """
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    xx = tf.get_variable(subkey)
                    session.run(tf.get_variable(subkey).assign(data))

# %%
def print_all_variables(train_only=True):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)

    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try:  # TF1.0
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

    # %%


# ***** the followings are just for test the tensor size at diferent layers *********##

# %%
def weight(kernel_shape, is_uniform=True):
    """ weight initializer
    Args:
        shape: the shape of weight
        is_uniform: boolen type.
                if True: use uniform distribution initializer
                if False: use normal distribution initizalizer
    Returns:
        weight tensor
    """
    w = tf.get_variable(name='weights',
                        shape=kernel_shape,
                        initializer=tf.contrib.layers.xavier_initializer())
    return w


# %%
def bias(bias_shape):
    """bias initializer
    """
    b = tf.get_variable(name='biases',
                        shape=bias_shape,
                        initializer=tf.constant_initializer(0.0))
    return b

# %%
