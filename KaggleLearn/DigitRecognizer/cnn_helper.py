import tensorflow as tf          # TensorFlow
import matplotlib.pyplot as plt  # matplotlib绘图
import numpy as np               # Numpy
from sklearn.metrics import confusion_matrix    # 混淆矩阵，分析模型误差

import time       # 计时
from datetime import timedelta
import math


def plot_images(images, cls_true, img_shape, cls_pred=None):
    """
    绘制图像，输出真实标签与预测标签
    images:   图像（9张）
    cls_true: 真实类别
    cls_pred: 预测类别
    """
    assert len(images) == len(cls_true) == 9   # 保证存在9张图片


    fig, axes = plt.subplots(3, 3)   # 创建3x3个子图的画布
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整每张图之间的间隔

    for i, ax in enumerate(axes.flat):
        # 绘图，将一维向量变为二维矩阵，黑白二值图像使用 binary
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:  # 如果未传入预测类别
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)

        # 删除坐标信息
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_example_errors(data_test, cls_pred, correct, img_shape):
    # 计算错误情况
    incorrect = (correct == False)
    images = data_test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data_test.cls[incorrect]

    # 随机挑选9个
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    indices = indices[:9]

    plot_images(images[indices], cls_true[indices], img_shape, cls_pred[indices])


def plot_confusion_matrix(cls_true, cls_pred):

    # 使用scikit-learn的confusion_matrix来计算混淆矩阵
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

    # 打印混淆矩阵
    print(cm)

    num_classes = cm.shape[0]

    # 将混淆矩阵输出为图像
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # 调整图像
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_conv_weights(weights, input_channel=0):
    # weights_conv1 or weights_conv2.


    # 获取权重最小值最大值，这将用户纠正整个图像的颜色密集度，来进行对比
    w_min = np.min(weights)
    w_max = np.max(weights)

    # 卷积核树木
    num_filters = weights.shape[3]

    # 每行需要输出的卷积核网格数
    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        # 只输出有用的子图.
        if i<num_filters:
            # 获得第i个卷积核在特定输入通道上的权重
            img = weights[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # 移除坐标.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_conv_layer(values):
    # layer_conv1 or layer_conv2

    # 卷积核数目
    num_filters = values.shape[3]

    # 每行需要输出的卷积核网格数
    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        # 只输出有用的子图.
        if i<num_filters:
            # 获取第i个卷积核的输出
            img = values[0, :, :, i]

            ax.imshow(img, interpolation='nearest', cmap='binary')

        # 移除坐标.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_image(image, img_shape):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')
    plt.show()