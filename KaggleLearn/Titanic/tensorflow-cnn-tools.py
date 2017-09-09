# coding: utf-8

# In[23]:


# Load in our libraries
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn import model_selection
import KaggleLearn.Titanic.testCnn as testCnn
import KaggleLearn.tools.tools as tools
warnings.filterwarnings('ignore')
# In[24]:


# Load in the train and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Store our passenger ID for easy access
PassengerId = test['PassengerId']

# In[25]:


train.head(3)

# 毫无疑问，我们的任务是以某种方式从分类变量中提取信息。
# 
# Well it is no surprise that our task is to somehow extract the information out of the categorical variables
# 
# **特征工程**
# 
# **Feature Engineering**
# 
# 在这里，必须扩展到Sina的特征工程理念，这是非常全面和周全的笔记，所以请看看他的工作
# 
# Here, credit must be extended to Sina's very comprehensive and well-thought
# out notebook for the feature engineering ideas so please check out his work
# 
# [Titanic Best Working Classfier](https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier): by Sina

# In[26]:


full_data = [train, test]

# Some features of my own that I have added in
# Gives the length of the name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Feature engineering steps taken from Sina
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# Create a New feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)


# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age']

# In[27]:


train.head(2)

# In[28]:


# 特征选择
# 准备训练和测试数据


print(train.shape)
y = train["Survived"]
print("y.shape", y.shape)

# print(train.shape)
tf_train = train.drop(["Survived"], axis=1)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
tf_train = tf_train.drop(drop_elements, axis=1)
tf_train = tf_train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
print(tf_train.shape)
# X_train_ss_new = SelectKBest(f_classif, k=16).fit_transform(X_train, y)
# print(X_train_ss_new.shape)
tf_train.head(2)

# X_train.to_csv('./out1.csv')

tf_train_new = SelectKBest(f_classif, k=9).fit_transform(tf_train, y)
# tf_train_new_pd = pd.DataFrame(tf_train_new)
# tf_train_new_pd.to_csv('./out2.csv')
tf_test = test[['Pclass', 'Sex', 'Parch', 'Fare', 'Embarked', 'Name_length', 'Has_Cabin', 'IsAlone', 'Title']]
# tf_test = test.drop(drop_elements, axis = 1)
# tf_test = tf_test.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
# 独热编码
y = (np.arange(2) == y[:, None]).astype(np.float32)
print("y.shape", y.shape)
print("tf_train_new.shape", tf_train_new.shape)
tf_labels = y
# 'Pclass','Sex','Parch','Fare','Embarked','Name_length','Has_Cabin','IsAlone','Title'


tf_test = preprocessing.Normalizer().fit_transform(tf_test)
tf_train_new = preprocessing.Normalizer().fit_transform(tf_train_new)
y = preprocessing.Normalizer().fit_transform(y)


# In[29]:


# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 25

DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# image number to output
IMAGE_TO_DISPLAY = 10

def accuracy(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor,
  """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + '/accuracy', accuracy)
    return accuracy


# 保存结果
def savecsv(test_prediction_np, filename="submission_tf_titanic.csv"):
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': test_prediction_np})
    StackingSubmission.to_csv(filename, index=False)

# 形成验证数据


train_dataset, valid_dataset = model_selection.train_test_split(tf_train_new, test_size=0.3, random_state=0)
train_labels, valid_labels = model_selection.train_test_split(y, test_size=0.3, random_state=0)
# print(train_dataset.shape)
# print(valid_dataset.shape)
# print(train_labels.shape)
# print(valid_labels.shape)


graphCNN = tf.Graph()
N_CLASSES = 2
IS_PRETRAIN = True
learning_rate = 0.01

with graphCNN.as_default():
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 9])  # 原始数据的维度：9
    ys = tf.placeholder(tf.float32, [None, 2])  # 输出数据为维度：2

    # keep_prob = tf.placeholder(tf.float32)  # dropout的比例
    #
    x_image = tf.reshape(xs, [-1, 3, 3, 1])  # 原始数据9变成二维图片3*3
    logits = testCnn.TestCnn(x_image, N_CLASSES, IS_PRETRAIN)

    # prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # 计算 predition与y 差距 所用方法很简单就是用 suare()平方,sum()求和,mean()平均值

    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits))

    # print("loss=", loss)
    loss = tools.loss(logits=logits, labels=ys)
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)
    # 0.01学习效率,minimize(loss)减小loss误差
    # train_step = tf.train.AdamOptimizer(0.03).minimize(cross_entropy)

    accuracy = accuracy(logits, ys)

# print("tf_train_new.shape", tf_train_new.shape)
# print("tf_labels.shape", tf_labels.shape)
# print("y.shape", y.shape)

print(type(tf_train_new))
print(type(y))


with tf.Session(graph=graphCNN) as sess:
    sess.run(tf.global_variables_initializer())
    summary_op = tf.summary.merge_all()
    train_log_dir = './logs/train/'
    val_log_dir = './logs/val/'

    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    # 训练500次
    for i in range(500):
        summary_t, _, tra_loss, tra_acc = sess.run([summary_op, train_op, loss, accuracy], feed_dict={xs: train_dataset, ys: train_labels})
        # 输出loss值
        if i % 50 == 0:
            # 每一百步保存

            summary_v, _, val_loss, val_acc = sess.run([summary_op, train_op, loss, accuracy],
                                                       feed_dict={xs: valid_dataset, ys: valid_labels})
            print('Step: %d, loss: %.4f, tra_accuracy: %.4f%%' % (i, tra_loss, tra_acc))
            print('Step: %d, loss: %.4f, val_accuracy: %.4f%%' % (i, val_loss, val_acc))
            tra_summary_writer.add_summary(summary_t, i)
            val_summary_writer.add_summary(summary_v, i)

    # 可视化
    prediction_value = sess.run(logits, feed_dict={xs: tf_test})
    #     print(prediction_value)
    prediction_value = np.argmax(prediction_value, 1)
    print(prediction_value.shape)
    #     print("test_prediction_np",test_prediction_np[0])
    #     print("test_prediction_np.shpe",test_prediction_np.shape)
    #     print("x.shpe",x.shape)
    #     print("x",x)
    #     print("test_prediction_np.shape",test_prediction_np.shape)
    savecsv(prediction_value, "fission_tf_2lcnn.csv")

    print("Complete")
