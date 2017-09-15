import KaggleLearn.Tools.input_data as inputData
import KaggleLearn.Model.VGG as VGG
import tensorflow as tf
import KaggleLearn.Tools.tools as tools
import KaggleLearn.DogVsCat.utils as utils
import numpy as np
import time

start = time.time()
# 定义参数
BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 224
IMG_H = 224
N_CLASSES = 1000
learning_rate = 0.01
MAX_STEP = 150
IS_PRETRAIN = False

pre_trained_weights = './VGG16/vgg16.npy'

with tf.name_scope('input'):
    img1 = utils.load_image("./test_data/tiger.jpeg")
    img2 = utils.load_image("./data/train/cat.0.jpg")

    batch1 = img1.reshape((1, 224, 224, 3))
    batch2 = img2.reshape((1, 224, 224, 3))

    batch = np.concatenate((batch1, batch2), 0)


    # 占位符定义输入数据形状和标签形状
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    # y = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    # loss = tools.loss(logits, y)
    # accuracy = tools.accuracy(logits, y)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    # train_op = tools.optimize(loss, learning_rate, my_global_step)
    # tf.global_variables() 获取程序中的变量
    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 读取训练好的VGG16
    # tools.load_with_skip(pre_trained_weights, sess, ['fc8'])
    tools.load(pre_trained_weights, sess)

    # 多线程
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    # # 记录图形和文件位置
    # tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    # val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    for step in np.arange(1):
        # summary_t, _, tra_loss, tra_acc = sess.run([summary_op, train_op, loss, accuracy],
        # _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
        #                                 feed_dict={x: tra_images})

        prob = sess.run([logits], feed_dict={x: batch})
        # print(prob[0][0])
        # print(prob[0][1])
        utils.print_prob(prob[0][0], './VGG16/synset.txt')
        utils.print_prob(prob[0][1], './VGG16/synset.txt')
        # print(prob[1])
        # print("logits", logits)
        # print("p", prob)
        # print("p。shape", len(prob[0]))
        print("prob.type", type(prob[0]))
        # utils.print_prob(prob[0], './VGG16/synset.txt')
        # utils.print_prob(prob[1], './VGG16/synset.txt')
        # 输出loss值
        # if step % 50 == 0:
        # 每一百步保存

        # summary_v, _, val_loss, val_acc = sess.run([summary_op, train_op, loss, accuracy],
        #                                            feed_dict={x: valid_dataset, y: valid_labels})
        # print('Step: %d, loss: %.4f, tra_accuracy: %.4f%%' % (step, tra_loss, tra_acc))
        # print('Step: %d, loss: %.4f, val_accuracy: %.4f%%' % (i, val_loss, val_acc))
        # tra_summary_writer.add_summary(summary_t, i)
        # val_summary_writer.add_summary(summary_v, i)

    end = time.time()
    print("time=", end - start)
    print("Complete")
