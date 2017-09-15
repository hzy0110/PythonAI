import KaggleLearn.Tools.input_data as inputData
import KaggleLearn.Model.VGG as VGG
import tensorflow as tf
import KaggleLearn.Tools.tools as tools
import numpy as np

# 定义参数
BATCH_SIZE = 64
CAPACITY = 256
IMG_W = 208
IMG_H = 208
N_CLASSES = 1000
learning_rate = 0.01
MAX_STEP = 150
IS_PRETRAIN = False
train_dir = "./data/train/"
test_dir = "./data/test/"
train_log_dir = './logs/train/'
val_log_dir = './logs/val/'
pre_trained_weights = './VGG16/vgg16.npy'

with tf.name_scope('input'):
    # 读取文件
    image_list, label_list = inputData.get_files(train_dir)
    # 转换数据文件
    image_batch, label_batch = inputData.get_batch(image_list, label_list, IMG_W, IMG_H, len(image_list), CAPACITY)
    label_batch_reshape = tf.reshape(label_batch, [-1, 1000])
    #
    # print(type(image_list[0]))
    # print(type(label_list[0]))
    # print(type(image_batch[0]))
    # print(type(label_batch[0]))
    # graphCNN = tf.Graph()
    # with graphCNN.as_default():
    # 占位符定义输入数据形状和标签形状
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y)
    accuracy = tools.accuracy(logits, y)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)
    # tf.global_variables() 获取程序中的变量
    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # with tf.Session(graph=graphCNN) as sess:
    #     sess.run(tf.global_variables_initializer())
    image_list_len = len(image_list)
    # summary_op = tf.summary.merge_all()

    # 读取训练好的VGG16
    # tools.load_with_skip(pre_trained_weights, sess, ['fc8'])
    tools.load(pre_trained_weights, sess)

    # 多线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    # # 记录图形和文件位置
    # tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    # val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    for step in np.arange(1):
        # 数据分批
        offset = (step * BATCH_SIZE) % (image_list_len - BATCH_SIZE)
        batch_data_train = image_batch[offset:(offset + BATCH_SIZE), :, :, :]
        batch_labels_train = label_batch_reshape[offset:(offset + BATCH_SIZE), :]
        # print("start")
        # print(batch_data_train.shape)
        # print(batch_labels_train.shape)
        # tra_images = batch_data_train.eval()
        # tra_images, batch_labels_train.eval()
        # print("end")
        tra_images, tra_labels = sess.run([batch_data_train, batch_labels_train])
        # print(type(tra_images))
        # print(type(tra_labels))
        print(tra_images.shape)
        print(tra_labels.shape)
        # summary_t, _, tra_loss, tra_acc = sess.run([summary_op, train_op, loss, accuracy],
        _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                        feed_dict={x: tra_images, y: tra_labels})
        # 输出loss值
        # if step % 50 == 0:
        # 每一百步保存

        # summary_v, _, val_loss, val_acc = sess.run([summary_op, train_op, loss, accuracy],
        #                                            feed_dict={x: valid_dataset, y: valid_labels})
        print('Step: %d, loss: %.4f, tra_accuracy: %.4f%%' % (step, tra_loss, tra_acc))
        # print('Step: %d, loss: %.4f, val_accuracy: %.4f%%' % (i, val_loss, val_acc))
        # tra_summary_writer.add_summary(summary_t, i)
        # val_summary_writer.add_summary(summary_v, i)

    print("Complete")
