import string, os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
from sklearn import neighbors, linear_model, svm

dir = 'genki4k/files'
print('----------- no sub dir')

# prepare the data
files = os.listdir(dir)
# for f in files:
    # print(dir + os.sep + f)

file_path = dir + os.sep + files[14]

print(file_path)

dic_mat = scipy.io.loadmat(file_path)

data_mat = dic_mat['Hog_Feat']

print('feature: ', data_mat.shape)

# print data_mat.dtype

file_path2 = dir + os.sep + files[15]

# print file_path2

dic_label = scipy.io.loadmat(file_path2)

label_mat = dic_label['Label']

file_path3 = dir + os.sep + files[16]

print('fiel 3 path: ', file_path3)

dic_T = scipy.io.loadmat(file_path3)

T = dic_T['T']
T = T - 1

print(T.shape)

label = label_mat.ravel()

# Acc=np.zeros((1,4))

Acc = [0, 0, 0, 0]

for i in range(0, 4):
    print("the fold %d" % (i + 1))
    train_ind = []
    for j in range(0, 4):
        if j == i:
            test_ind = T[j]
        else:
            train_ind.extend(T[j])
        #    print len(test_ind), len(train_ind)
        #    print max(test_ind), max(train_ind)
    train_x = data_mat[train_ind, :]
    test_x = data_mat[test_ind, :]
    train_y = label[train_ind]
    test_y = label[test_ind]
    #   SVM
    clf = svm.LinearSVC()
    #   KNN
    #    clf = neighbors.KNeighborsClassifier(n_neighbors=15)
    #    Logistic regression
    #    clf = linear_model.LogisticRegression()

    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    Acc[i] = np.mean(predict_y == test_y)
    print("Accuracy: %.2f" % (Acc[i]))

print("The mean average classification accuracy: %.2f" % (np.mean(Acc)))
