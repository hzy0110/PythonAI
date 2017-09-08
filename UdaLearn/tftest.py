# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
from six.moves import cPickle as pickle

from sklearn.linear_model import LogisticRegression
import numpy as np

size = 1000
image_size = 28
# loading data
print('loading data')
with open('notMNIST.pickle', 'rb') as f:
    data = pickle.load(f)
print('finish loading')
train_dt = data['train_dataset']
length = train_dt.shape[0]
train_dt = train_dt.reshape(length, image_size * image_size)
train_lb = data['train_labels']
test_dt = data['test_dataset']
length = test_dt.shape[0]
test_lb = data['test_labels']
test_dt = test_dt.reshape(length, image_size * image_size)
print('feature: ',  train_dt.shape)


def train_linear_logistic(tdata, tlabel):
    model = LogisticRegression(penalty='l2')
    # model = LogisticRegression()

    print('initializing model size is = {}'.format(size))
    print(len(tdata[:size, :]))
    print(len(tdata[:size, :][0]))
    print(tdata[:size, :])
    print(tlabel[:size])
    model.fit(tdata[:size], tlabel[:size])

    # print('testing model')
    # y_out = model.predict(test_dt)

    # print('the accurace of the mode of size = {} is {}'.format(size, np.sum(y_out == test_lb)*1.0/len(y_out)))

    return None


# print(train_dt)
train_linear_logistic(train_dt, train_lb)
# size = 100
# train_linear_logistic(train_dt, train_lb)
# size = 1000
# train_linear_logistic(train_dt, train_lb)
# size = 5000
# train_linear_logistic(train_dt, train_lb)
