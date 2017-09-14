import KaggleLearn.Tools.input_data as inputData
import KaggleLearn.Model.VGG as VGG
import tensorflow as tf
import KaggleLearn.Tools.tools as tools
import numpy as np

BATCH_SIZE = 64
CAPACITY = 256
IMG_W = 208
IMG_H = 208
N_CLASSES = 2
learning_rate = 0.01
MAX_STEP = 150
IS_PRETRAIN = False

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
y = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

with tf.Session() as sess:
    with open('./VGG16/vgg16-20160129.tfmodel', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tools.print_all_variables()
        # output = tf.import_graph_def(graph_def, input_map={'input:0':4.}, return_elements=['out:0'], name='a')
        # print(sess.run(x))


