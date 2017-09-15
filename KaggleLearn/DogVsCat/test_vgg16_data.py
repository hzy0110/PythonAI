import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import KaggleLearn.Tools.input_data as inputData
import KaggleLearn.DogVsCat.vgg16 as vgg16
import KaggleLearn.DogVsCat.utils as utils
import KaggleLearn.Tools.tools as tools

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./data/train/cat.0.jpg")
train_dir = "./data/train/"
test_dir = "./data/test/"
IMG_W = 224
IMG_H = 224
BATCH_SIZE = 64
CAPACITY = 256
N_CLASSES = 1000
batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        # image_list, label_list = inputData.get_files(train_dir)
        # # 转换数据文件
        # image_batch, label_batch = inputData.get_batch(image_list, label_list, IMG_W, IMG_H, len(image_list), CAPACITY)
        # label_batch_reshape = tf.reshape(label_batch, [-1, 1000])
        #
        # x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
        # y = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])
        # image_list_len = len(image_list)
        #
        # # 多线程
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in np.arange(1):
            # 数据分批
            # offset = (step * BATCH_SIZE) % (image_list_len - BATCH_SIZE)
            # batch_data_train = image_batch[offset:(offset + BATCH_SIZE), :, :, :]
            # batch_labels_train = label_batch_reshape[offset:(offset + BATCH_SIZE), :]
            # tra_images, tra_labels = sess.run([batch_data_train, batch_labels_train])
            images = tf.placeholder("float", [2, 224, 224, 3])
            feed_dict = {images: batch}

            vgg = vgg16.Vgg16()
            print(type(vgg))
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            # 打印训练过程图片
            n_feature = int(vgg.conv1_1.get_shape()[-1])

            prob = sess.run(vgg.prob, feed_dict=feed_dict)


            # print("prob.shape", prob.shape)
            # print("prob.type", type(prob))
            # print("vgg.prob", vgg.prob)
            # # print("prob=", prob)
            # utils.print_prob(prob[0], './VGG16/synset.txt')
            # utils.print_prob(prob[1], './VGG16/synset.txt')
