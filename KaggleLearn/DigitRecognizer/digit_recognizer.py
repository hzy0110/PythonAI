import numpy as np
import pandas as pd #数据分析
import operator
import csv
import time
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler


image_size = 28  # 宽和高的像素.
pixel_depth = 255.0  # 每个像素的深度，0-255
image_vector_length = 784  # 图片向量长度
# file_root = "/Users/hzy/code/PycharmProjects/KaggleLearn/DigitRecognizer/"

start_time = time.time()

#
# def toInt(array):
#     array = np.mat(array)
#     m, n = np.shape(array)
#     newArray = np.zeros((m, n))
#     for i in range(m):
#         for j in range(n):
#             newArray[i, j] = int(array[i, j])
#     return newArray
#
#
# def normalization(array):
#     """
#     归一化，数据范围-1 - 1之间
#     :param array:
#     :return:
#     """
#     # array = (array - pixel_depth / 2) / pixel_depth
#     array = array / pixel_depth
#     return array


def loadTrainData():
    l = []
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    l.remove(l[0])
    l = np.array(l)
    label = l[:, 0]
    data = l[:, 1:]
    # print(type(data))
    # print(data.dtype)
    # print(label.dtype)
    data = data.astype(float)
    label = label.astype(float)
    # print(data.dtype)
    # print(label.dtype)
    file.close()
    # 区间缩放，返回值为缩放到[0, 1]区间的数据
    return MinMaxScaler().fit_transform(data), label  # label 1*42000  data 42000*784


def loadTestData():
    l = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
            # 28001*784
    l.remove(l[0])
    # print(l[0])
    data = np.array(l)
    file.close()
    data = data.astype(float)
    return MinMaxScaler().fit_transform(data)  # data 28000*784


def loadTestResult():
    l = []
    with open('knn_benchmark.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
            # 28001*2
    l.remove(l[0])
    label = np.array(l)
    label = label.astype(float)
    # print(label)
    return label[:, 1]  # label 28000*1


def save2pickle(trainData, trainLabel, testData, testLabel):
    pickle_file = "dr-non-negative.pickle"
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': trainData,
            'train_labels': trainLabel,
            'test_dataset': testData,
            'test_labels': testLabel,
        }
        #     保存文件，pickle.HIGHEST_PROTOCOL是高压缩
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('不能保存文件', pickle_file, ':', e)
        raise


trainData, trainLabel = loadTrainData()
testData = loadTestData()
testLabel = loadTestResult()
save2pickle(trainData, trainLabel, testData, testLabel)
print('数据加载时间 %fs!' % (time.time() - start_time))
# print(trainData.shape)  # (42000, 784)
# print(testData.shape)  # (28000, 784)


# Multinomial Naive Bayes Classifier
# 需要使用非负特征值
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


test_classifiers = [
    # 'NB',
    # 'KNN',
    # 'LR',
    'RF',
    # 'DT',
    'SVM',
    'GBDT'
]

classifiers = {
    'NB': naive_bayes_classifier,
    'KNN': knn_classifier,
    'LR': logistic_regression_classifier,  # 逻辑回归
    'RF': random_forest_classifier,
    'DT': decision_tree_classifier,
    'SVM': svm_classifier,
    'SVMCV': svm_cross_validation,
    'GBDT': gradient_boosting_classifier
}

model_save_file = "classifier.pkl"
model_save = {}


def classifier():
    """
    分类
    :return:
    """
    with open('dr-non-negative.pickle', 'rb') as f:
        data = pickle.load(f)

    trainData = data['train_dataset']
    trainLabel = data['train_labels']
    testData = data['test_dataset']
    testLabel = data['test_labels']

    num_train, num_feat = trainData.shape
    num_test, num_feat = testData.shape
    print('******************** Data Info *********************')
    print('#训练数据集大小: %d, #测试数据集大小: %d, 维度: %d' % (num_train, num_test, num_feat))

    # 获取启动时间
    start_time = time.time()
    classifier_result = {}
    print(type(classifier_result))
    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()

        # 开始学习
        model = classifiers[classifier](trainData[:500], trainLabel[:500])
        model_time = time.time() - start_time
        # print('训练模型时间 %fs!' % (model_time))
        # 预测测试集合
        start_time = time.time()
        predict = model.predict(testData)
        predict_time = time.time() - start_time
        # print('predict时间 %fs!' % (predict_time))
        # 模型保存
        if model_save_file is not None:
            model_save[classifier] = model
        # if is_binary_class:
        #     precision = metrics.precision_score(test_y, predict)
        #     recall = metrics.recall_score(test_y, predict)
        #     print('精准率: %.2f%%, 召回率: %.2f%%' % (100 * precision, 100 * recall))
        # 测试准确性
        accuracy = metrics.accuracy_score(testLabel, predict)
        # print('准确性: %.2f%%' % (100 * accuracy))
        classifier_result[classifier] = ("训练时间" + str(model_time), "预测时间" + str(predict_time), "准确性: " + str(100 * accuracy))
    print(classifier_result)
    if model_save_file is not None:
        pickle.dump(model_save, open(model_save_file, 'wb'))
    # 模型保存和加载
    # pickle.dump(model, open('bayes.pk', 'wb'))
    # clf2 = pickle.load(open('bayes.pk', 'rb'))
    # print(type(clf2))
    return predict


def saveResult(result):
    np.savetxt('submission_softmax.csv',
               np.c_[range(1, len(result) + 1), result],
               delimiter=',',
               header='ImageId,Label',
               comments='',
               fmt='%d')


predict = classifier()

# saveResult(predict)


def rePredict(monel_file):
    with open('dr-non-negative.pickle', 'rb') as f:
        data = pickle.load(f)

    testData = data['test_dataset']
    testLabel = data['test_labels']

    start_time = time.time()
    clf_dict = pickle.load(open(monel_file, 'rb'))
    predict = clf_dict["SVM"].predict(testData)

    accuracy = metrics.accuracy_score(testLabel, predict)
    # print('准确性: %.2f%%' % (100 * accuracy))
    td = time.time() - start_time
    print("训练时间" + str(td), "准确性: " + str(100 * accuracy))

    # for clf in clf_dict:

        # print(type(clf))

# rePredict(model_save_file)


# 92.14
