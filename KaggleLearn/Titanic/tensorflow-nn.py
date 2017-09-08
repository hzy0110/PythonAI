# coding: utf-8

# In[124]:


# Load in our libraries
import re
import numpy as np
import pandas as pd
import tensorflow as tf

import warnings

warnings.filterwarnings('ignore')


# Load in the train and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Store our passenger ID for easy access
PassengerId = test['PassengerId']

# In[126]:


train.head(3)



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
    dataset.loc[dataset['Age'] > 64, 'Age'];

# In[128]:


train.head(2)

# In[129]:


# 特征选择
# 准备训练和测试数据
from sklearn.feature_selection import SelectKBest, f_classif

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
# tf_test = test[['Pclass','Sex','Parch','Fare','Embarked','Name_length','Has_Cabin','IsAlone','Title']]
tf_test = test.drop(drop_elements, axis=1)
# tf_test = tf_test.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
# 独热编码
y = (np.arange(2) == y[:, None]).astype(np.float32)
print("y.shape", y.shape)
print("tf_train_new.shape", tf_train_new.shape)
# 'Pclass','Sex','Parch','Fare','Embarked','Name_length','Has_Cabin','IsAlone','Title'



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



# 计算精准度
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


# 保存结果
def savecsv(test_prediction_np, filename="submission_tf_titanic.csv"):
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': test_prediction_np})
    StackingSubmission.to_csv(filename, index=False)


# In[137]:


# 形成验证数据
from sklearn import model_selection

train_dataset, valid_dataset = model_selection.train_test_split(tf_train, test_size=0.3, random_state=0)
train_labels, valid_labels = model_selection.train_test_split(y, test_size=0.3, random_state=0)
print(train_dataset.shape)
print(valid_dataset.shape)
print(train_labels.shape)
print(valid_labels.shape)

# In[138]:


# 根据tf任务3.1加入l2 0.77990
# 加入DropOut 0.78947
# 加入梯度学习率，原学习率0.5  0.78469
# 加入5层隐藏层 0.78469
# 减少到3层，初始化节点减小到256

batch_size = 64
regularation_param = 0.0001
keep_prob = 0.8
graph = tf.Graph()
num_labels = 2
hidden_nodes = 512

print(train_dataset.shape)


def compute_logits(data, weightss, biasess, dropout_vals=None):
    temp = data
    if dropout_vals:
        for w, b, d in zip(weightss[:-1], biasess[:-1], dropout_vals[:-1]):
            temp = tf.nn.relu_layer(tf.nn.dropout(temp, d), w, b)
        temp = tf.matmul(temp, weightss[-1]) + biasess[-1]
    else:
        for w, b in zip(weightss[:-1], biasess[:-1]):
            temp = tf.nn.relu_layer(temp, w, b)
        temp = tf.matmul(temp, weightss[-1]) + biasess[-1]
    return temp


with graph.as_default():
    # -----------------------------------------1
    # 输入 
    # placeholder 插入一个待初始化的张量占位符
    # 重要事项：这个张量被求值时会产生错误。 
    # 它的值必须在Session.run(), Tensor.eval() 或 Operation.run() 中使用feed_dict的这个可选参数来填充。
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, train_dataset.shape[1]))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    # 创建一个常量张量
    # tf_valid_dataset = Tensor("Const:0", shape=(10000, 784), dtype=float32)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(tf_test)

    # 转数据类型，f64->f32
    tf_train_dataset = tf.to_float(tf_train_dataset)
    tf_valid_dataset = tf.to_float(tf_valid_dataset)
    tf_test_dataset = tf.to_float(tf_test_dataset)

    # 变量
    # 梯度学习率
    # 初始的学习速率
    starter_learning_rate = 0.03
    # 全局的step，与 decay_step 和 decay_rate一起决定了 learning rate的变化
    global_step = tf.Variable(0, trainable=False)
    # 衰减速度
    decay_steps = 50
    # 衰减系数
    decay_rate = 0.8
    # 如果staircase=True，那就表明每decay_steps次计算学习速率变化，更新原始学习速率.
    # 如果是False，那就是每一步都更新学习速率
    staircase = False
    # 指数衰减:法通过这个函数，可以先使用较大的学习率来快速得到一个比较优的解，然后随着迭代的继续逐步减小学习率，使得模型在训练后期更加稳定
    # 87.7% 仅仅指数衰减
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase)

    # 当你训练一个模型的时候，你使用变量去保存和更新参数。
    # 在Tensorflow中变量是内存缓冲区中保存的张量（tensor）

    # 第一层
    # truncated_normal 从一个正态分布片段中输出随机数值,
    # 生成的值会遵循一个指定了平均值和标准差的正态分布，只保留两个标准差以内的值，超出的值会被弃掉重新生成。
    # 返回 一个指定形状并用正态分布片段的随机值填充的张量
    # 数字平方根
    x = 2.0
    weights1 = tf.Variable(
        tf.truncated_normal([train_dataset.shape[1], hidden_nodes], stddev=np.sqrt(x / hidden_nodes)))
    biases1 = tf.Variable(tf.zeros([hidden_nodes]))

    # 第二层
    weights2 = tf.Variable(
        tf.truncated_normal([hidden_nodes, int(hidden_nodes / 2)], stddev=np.sqrt(x / hidden_nodes / 2)))
    biases2 = tf.Variable(tf.zeros([hidden_nodes / 2]))
    hidden_nodes = int(hidden_nodes / 2)

    # 第三层
    weights3 = tf.Variable(
        tf.truncated_normal([hidden_nodes, int(hidden_nodes / 2)], stddev=np.sqrt(x / hidden_nodes / 2)))
    biases3 = tf.Variable(tf.zeros([hidden_nodes / 2]))
    hidden_nodes = int(hidden_nodes / 2)

    # 第四层 94.5
    weights4 = tf.Variable(
        tf.truncated_normal([hidden_nodes, int(hidden_nodes / 2)], stddev=np.sqrt(x / hidden_nodes / 2)))
    biases4 = tf.Variable(tf.zeros([hidden_nodes / 2]))
    hidden_nodes = int(hidden_nodes / 2)

    # 第五层 94.5
    weights5 = tf.Variable(
        tf.truncated_normal([hidden_nodes, int(hidden_nodes / 2)], stddev=np.sqrt(x / hidden_nodes / 2)))
    biases5 = tf.Variable(tf.zeros([hidden_nodes / 2]))
    hidden_nodes = int(hidden_nodes / 2)

    weights_end = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
    biases_end = tf.Variable(tf.zeros([num_labels]))

    print("weights_end.shape", weights_end.shape)
    print("biases_end.shape", biases_end.shape)
    print(train_dataset.shape)

    # DropOut
    drop = tf.nn.dropout(tf_train_dataset, keep_prob=keep_prob)

    # 训练计算.
    #     train_logits = tf.add(tf.matmul(drop, weights), biases)
    #     valid_logits = tf.add(tf.matmul(tf_valid_dataset, weights), biases)
    #     test_logits = tf.add(tf.matmul(tf_test_dataset, weights), biases)
    train_logits = compute_logits(tf_train_dataset, [weights1, weights2, weights3, weights4, weights5, weights_end],
                                  [biases1, biases2, biases3, biases4, biases5, biases_end],
                                  dropout_vals=(1.0, 1, 1, 1, 1, 1.0))
    valid_logits = compute_logits(tf_valid_dataset, [weights1, weights2, weights3, weights4, weights5, weights_end],
                                  [biases1, biases2, biases3, biases4, biases5, biases_end])
    test_logits = compute_logits(tf_test_dataset, [weights1, weights2, weights3, weights4, weights5, weights_end],
                                 [biases1, biases2, biases3, biases4, biases5, biases_end])

    # 加l2_loss
    hpl2 = regularation_param * (tf.nn.l2_loss(weights1)
                                 + tf.nn.l2_loss(weights2)
                                 + tf.nn.l2_loss(weights3)
                                 + tf.nn.l2_loss(weights4)
                                 + tf.nn.l2_loss(weights5)
                                 )
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=train_logits))
    loss = tf.add(loss, hpl2)

    # 最优化.因为深度学习常见的是对于梯度的优化，也就是说，优化器最后其实就是各种对于梯度下降算法的优化。
    #     optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(loss, global_step=global_step)

    train_prediction = tf.nn.softmax(train_logits)
    valid_prediction = tf.nn.softmax(valid_logits)
    test_prediction = tf.nn.softmax(test_logits)

# In[139]:


num_steps = 10000
print(type(train_dataset))
print(type(train_labels))
train_dataset_np = train_dataset.values
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # 在训练数据中选择一个已被随机化的偏移量.
        # 提醒: 我们能使用更好的随机化穿过所有数据.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        #         print(train_dataset.shape)
        #         print(train_labels.shape)
        # 生成一个小批量数据
        batch_data = train_dataset_np[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        #         print("batch_data.shape",batch_data.shape)
        #         print("batch_labels.shape",batch_labels.shape)
        # feed_dict的作用是给使用placeholder创建出来的tensor赋值。
        # 其实，他的作用更加广泛：feed 使用一个 值临时替换一个 op 的输出结果. 
        # 你可以提供 feed 数据作为 run() 调用的参数. feed 只在调用它的方法内有效, 方法结束, feed 就会消失.
        #  传递值到tf的命名空间  
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        summary, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

            # 获取结果，用于保存
    test_prediction_np = test_prediction.eval()
    test_prediction_np = np.argmax(test_prediction_np, 1)
    #     print("test_prediction_np.shape",test_prediction_np.shape)
    #     print("test_prediction_np",test_prediction_np)
    savecsv(test_prediction_np, "submission_tf_2lnn.csv")


# In[ ]:
