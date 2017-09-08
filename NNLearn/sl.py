from sklearn import svm
import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import pickle


def load_samples(dataset="training_data"):
    image, label = load_mnist(dataset)

    # 把28*28二维数据转为一维数据
    X = [np.reshape(x, (28 * 28)) for x in image]
    X = [x / 255.0 for x in X]  # 灰度值范围(0-255)，转换为(0-1)
    # print(X.shape)

    pair = list(zip(X, label))
    return pair


def load_mnist(dataset="training_data", digits=np.arange(10), path="."):
    if dataset == "training_data":
        fname_image = os.path.join(path, 'MNIST_data/train-images-idx3-ubyte')
        fname_label = os.path.join(path, 'MNIST_data/train-labels-idx1-ubyte')
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 'MNIST_data/t10k-images-idx3-ubyte')
        fname_label = os.path.join(path, 'MNIST_data/t10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")

    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

if __name__ == '__main__':

    train_set = load_samples(dataset='training_data')
    test_set = load_samples(dataset='testing_data')

    train_X = []
    train_Y = []
    for feature in train_set:
        train_X.append(feature[0])
        train_Y.append(feature[1][0])

    clf = svm.SVR()
    clf.fit(train_X, train_Y)  # 很耗时(我吃完饭回来，还没完，蛋碎... i5 CPU-8G RAM)

    # with open('minst.module', 'wb') as f:
    # pickle.dump(clf, f)

    # with open('minst.module', 'rb') as f:
    #	clf = pickle.load(f)

    test_X = []
    test_Y = []
    for feature in test_set:
        test_X.append(feature[0])
        test_Y.append(feature[1][0])

    # 准确率
    correct = 0
    i = 0
    for feature in test_X:
        predict = clf.predict(np.array(feature).reshape(1, -1))
        if round(float(predict)) == test_Y[i]:
            correct += 1
        i = i + 1
    print("准确率: ", correct / len(test_X))