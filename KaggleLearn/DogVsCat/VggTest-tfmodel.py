import KaggleLearn.Tools.input_data as inputData
import KaggleLearn.Model.VGG as VGG
import KaggleLearn.DogVsCat.vgg161 as vgg16
import tensorflow as tf
import KaggleLearn.Tools.tools as tools
import KaggleLearn.DogVsCat.utils as utils
import numpy as np
import time
from scipy.misc import imread, imresize
from KaggleLearn.DogVsCat.imagenet_classes import class_names
import os

start = time.time()
# 定义参数
BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 224
IMG_H = 224
N_CLASSES = 1000
learning_rate = 0.01
MAX_STEP = 150
IS_PRETRAIN = True

pre_trained_weights = './VGG16/vgg16.npy'
pre_trained_weights_npz = './VGG16/vgg16_weights.npz'
test_dir = "./data/test/"
with tf.name_scope('input'):

    # imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # img1 = imread('./data/train/cat.1.jpg', mode='RGB')
    # img1 = imresize(img1, (224, 224))
    # # logits = VGG.convlayers(img1)
    # # logits = VGG.VGG16(imgs, N_CLASSES, IS_PRETRAIN)
    # logits = VGG.VGG16N(imgs, N_CLASSES, IS_PRETRAIN)
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    #
    # # 读取训练好的VGG16，不同文件有影响
    # # tools.load_with_skip(pre_trained_weights, sess, ['fc8'])
    # tools.loadnpz(pre_trained_weights_npz, sess)
    # # tools.load(pre_trained_weights, sess)
    #
    # prob = sess.run(logits, feed_dict={imgs: [img1]})[0]
    # preds = (np.argsort(prob)[::-1])[0:5]
    # for p in preds:
    #     print(class_names[p], prob[p])

    #############分割############
    # 用utils.load_image分析数据会出问题
    # img1 = utils.load_image("./data/train/cat.1.jpg")
    # img2 = utils.load_image("./test_data/tiger.jpeg")

    test_file = inputData.get_files_Test(test_dir)
    print("test_file.shape", len(test_file))
    # print("type(test_file)", type(test_file))
    # print("test_file.shape", test_file.shape)
    test_images = [test_dir + i for i in os.listdir(test_dir)]
    print("test_images.shape", len(test_images))
    batch = np.concatenate(test_images, 0)
    # img1 = imread('./data/train/cat.1.jpg', mode='RGB')
    # img2 = imread('./test_data/tiger.jpeg', mode='RGB')
    # img1 = imresize(img1, (224, 224))
    # img2 = imresize(img2, (224, 224))
    # img1 = img1.reshape((1, 224, 224, 3))
    # img2 = img2.reshape((1, 224, 224, 3))
    # batch = np.concatenate((img1, img2), 0)
    # print("batch.type", type(batch))
    # print("img1.shape", img1.shape)
    # print("img1.type", type(img1))
    # print("img2.shape", img2.shape)
    # print("img2.type", type(img2))
    print("batch.shape", batch.shape)

    images = tf.placeholder("float", [None, 224, 224, 3])

    # logits = VGG.convlayers(images)
    logits = VGG.VGG16N(images, N_CLASSES, IS_PRETRAIN)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    #
    # 读取训练好的VGG16，不同文件，会有影响
    # tools.load_with_skip(pre_trained_weights, sess, ['fc8'])
    tools.loadnpz(pre_trained_weights_npz, sess)
    # tools.load(pre_trained_weights, sess)
    # tools.load_weights(pre_trained_weights, sess)

    # prob = sess.run(logits, feed_dict={images: [img1]})[0]
    # preds = (np.argsort(prob)[::-1])[0:5]
    # for p in preds:
    #     print(class_names[p], prob[p])

    prob = sess.run(logits, feed_dict={images: batch})
    print(prob)
    print("prob.shape", prob.shape)
    print("prob.type", type(prob[0]))
    # print(prob[0][1])
    utils.print_prob(prob[0], './VGG16/synset.txt')
    utils.print_prob(prob[1], './VGG16/synset.txt')
    # tools.print_all_variables()
    end = time.time()
    print("time=", end - start)
    print("Complete")
