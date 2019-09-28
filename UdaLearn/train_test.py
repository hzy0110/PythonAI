# !usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
from sklearn import metrics
import numpy as np
import pickle as pickle
# from data_util import DataUtils

# from imp import reload
# reload(sys)
# sys.setdefaultencoding('utf8')

import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf


# print("111111111")
# print(type(mnist.train)) 
# x = mnist.train
# x = mnist.train


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f, encoding='bytes')
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y


def read_data_dr():
    with open('dr.pickle', 'rb') as f:
        # print(f)
        data = pickle.load(f)
    # train, val, test = pickle.load(f)
    f.close()
    train_x = data['train_dataset']
    train_y = data['train_labels']
    test_x = data['test_dataset']
    test_y = data['test_labels']
    return train_x, train_y, test_x, test_y


def read_data_notMNIST():
    with open('notMNIST.pickle', 'rb') as f:
        # print(f)
        data = pickle.load(f)
    # train, val, test = pickle.load(f)
    f.close()
    train_x = data['train_dataset']
    train_y = data['train_labels']
    test_x = data['test_dataset']
    test_y = data['test_labels']
    return train_x, train_y, test_x, test_y

def read_data_MNIST():
    # one_hot=True会导致train_y的数据结构是one_hot的形式
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels
    return train_x, train_y, test_x, test_y
if __name__ == '__main__':
    data_file = "mnist.pkl.gz"
    # data_file = "notMNIST.pickle"
    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = [  # 'NB',
        # 'KNN',
        'LR',
        # 'RF',
        # 'DT',
        # 'SVM',
        # 'GBDT'
    ]
    classifiers = {  # 'NB': naive_bayes_classifier,
        # 'KNN': knn_classifier,
        'LR': logistic_regression_classifier,#逻辑分类
        # 'RF': random_forest_classifier,
        # 'DT': decision_tree_classifier,
        # 'SVM': svm_classifier,
        # 'SVMCV': svm_cross_validation,
        # 'GBDT': gradient_boosting_classifier
    }

    print('reading training and testing data...')
    # train_x, train_y, test_x, test_y = read_data(data_file)
    # print(test_y.shape)

    # train_x, train_y, test_x, test_y = read_data_notMNIST()
    # print(test_y.shape)
    # train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[1])
    # test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[1])

    # train_x, train_y, test_x, test_y = read_data_MNIST()
    # print(test_y.shape)
    # print(train_y.shape)
    # print(train_y[:10])
    # print(type(train_y[:10]))

    train_x, train_y, test_x, test_y = read_data_dr()

    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape

    # print(type(train_x))

    is_binary_class = (len(np.unique(train_y)) == 2)
    print('******************** Data Info *********************')
    print('#训练数据集大小: %d, #测试数据集大小: %d, 维度: %d' % (num_train, num_test, num_feat))

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)

        print('训练时间 %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        print(predict)
        if model_save_file is not None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print('精准率: %.2f%%, 召回率: %.2f%%' % (100 * precision, 100 * recall))

        accuracy = metrics.accuracy_score(test_y, predict)
        print('准确性: %.2f%%' % (100 * accuracy))

    if model_save_file is not None:
        pickle.dump(model_save, open(model_save_file, 'wb'))
