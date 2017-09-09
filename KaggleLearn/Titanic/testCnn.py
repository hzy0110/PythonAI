import tensorflow as tf
import KaggleLearn.tools.tools as tools


def TestCnn(x, n_classes, is_retrain=True):
    with tf.name_scope('CNN'):
        x = tools.conv('conv1_1', x, 64, kernel_size=[2, 2], stride=[1, 1, 1, 1], is_pretrain=is_retrain)
        with tf.name_scope('pool1'):
            x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        x = tools.conv('conv2_1', x, 64, kernel_size=[2, 2], stride=[1, 1, 1, 1], is_pretrain=is_retrain)
        with tf.name_scope('pool2'):
            x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.FC_layer('fc3', x, out_nodes=4096)
        with tf.name_scope('batch_norm1'):
            x = tools.batch_norm(x)
        x = tools.FC_layer('fc4', x, out_nodes=n_classes)
        return x

