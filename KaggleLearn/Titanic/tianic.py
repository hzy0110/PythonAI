import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC
from sklearn import metrics

data_train = pd.read_csv("train.csv")


# print(type(data_train))


# PassengerId => 乘客ID
# Pclass => 乘客等级(1/2/3等舱位)
# Name => 乘客姓名
# Sex => 性别
# Age => 年龄
# SibSp => 堂兄弟/妹个数
# Parch => 父母与小孩个数
# Ticket => 船票信息
# Fare => 票价
# Cabin => 客舱
# Embarked => 登船港口

def plt1():
    # 画些图来看看属性和结果之间的关系
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
    data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
    plt.title(u"获救情况 (1为获救)")  # 标题
    plt.ylabel(u"人数")

    plt.subplot2grid((2, 3), (0, 1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel(u"人数")
    plt.title(u"乘客等级分布")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data_train.Survived, data_train.Age)
    plt.ylabel(u"年龄")  # 设定纵坐标名称
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"按年龄看获救分布 (1为获救)")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")  # plots an axis lable
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')  # sets our legend for our graph.

    plt.subplot2grid((2, 3), (1, 2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")
    plt.show()


def plt2():
    # 看看各乘客等级的获救情况
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各乘客等级的获救情况")
    plt.xlabel(u"乘客等级")
    plt.ylabel(u"人数")
    plt.show()


def plt3():
    # 看看各性别的获救情况
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
    Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
    df = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
    df.plot(kind='bar', stacked=True)
    plt.title(u"按性别看获救情况")
    plt.xlabel(u"性别")
    plt.ylabel(u"人数")
    plt.show()


def plt4():
    # 然后我们再来看看各种舱级别情况下各性别的获救情况
    fig = plt.figure()
    fig.set(alpha=0.65)  # 设置图像透明度，无所谓
    plt.title(u"根据舱等级和性别的获救情况")

    ax1 = fig.add_subplot(141)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(
        kind='bar', label="female highclass", color='#FA2479')
    ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
    ax1.legend([u"女性/高级舱"], loc='best')

    ax2 = fig.add_subplot(142, sharey=ax1)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(
        kind='bar', label='female, low class', color='pink')
    ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"女性/低级舱"], loc='best')

    ax3 = fig.add_subplot(143, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(
        kind='bar', label='male, high class', color='lightblue')
    ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/高级舱"], loc='best')

    ax4 = fig.add_subplot(144, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(
        kind='bar', label='male low class', color='steelblue')
    ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/低级舱"], loc='best')

    plt.show()


def plt5():
    # 我们看看各登船港口的获救情况
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各登录港口乘客的获救情况")
    plt.xlabel(u"登录港口")
    plt.ylabel(u"人数")

    plt.show()


def df1():
    # 堂兄弟/妹，孩子/父母有几人，对是否获救的影响
    g = data_train.groupby(['SibSp', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    print(df)

    g = data_train.groupby(['SibSp', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    print(df)


def plt6():
    # 按Cabin有无看获救情况
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
    Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
    df = pd.DataFrame({u'有': Survived_cabin, u'无': Survived_nocabin}).transpose()
    df.plot(kind='bar', stacked=True)
    plt.title(u"按Cabin有无看获救情况")
    plt.xlabel(u"Cabin有无")
    plt.ylabel(u"人数")
    plt.show()


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
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


test_classifiers = [
    # 'NB',
    'KNN',
    'LR',
    'RF',
    'DT',
    'SVM',
    'SVMCV',
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

# 数据准备开始
# 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def dummies(data_train):
    """
    把数据维度展平，以Cabin为例，原本一个属性维度，因为其取值可以是[‘yes’,’no’]，而将其平展开为’Cabin_yes’,’Cabin_no’两个属性
    :param data_train:
    :return:
    """
    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')

    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')

    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')

    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


def scaler(scaler_df):
    # 缩放到0-1之间
    # 标准化
    scaler_df['Age_scaled'] = preprocessing.scale(scaler_df['Age'])
    scaler_df['Fare_scaled'] = preprocessing.scale(scaler_df['Fare'])
    return scaler_df


def pretreatment_test():
    """
    读取测试数据集
    :return:
    """
    data_test = pd.read_csv("test.csv")
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    # 接着我们对test_data做和train_data中一致的特征变换
    # 首先用同样的RandomForestRegressor模型填上丢失的年龄
    tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

    data_test = set_Cabin_type(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    df_test['Age_scaled'] = preprocessing.scale(df_test['Age'])
    df_test['Fare_scaled'] = preprocessing.scale(df_test['Fare'])
    return df_test

# 使用随机森林填补年龄
data_train, rfr = set_missing_ages(data_train)
# 处理Cabin空值和非空
data_train = set_Cabin_type(data_train)
# 把数据项展平
data_train = dummies(data_train)
# 数据标准化
data_train = scaler(data_train)
# 处理测试数据
data_test = pretreatment_test()
test = data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 数据准备结束


def lr(dt):
    # 用正则取出我们要的属性值
    train_df = dt.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到RandomForestRegressor之中
    # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # clf.fit
    clf = gradient_boosting_classifier(X, y)

    # 评测测试数据集
    predictions = clf.predict(test)

    return clf, train_df, predictions

# 模型化并测试数据
clf, data_train, predictions = lr(data_train)


def saveResult(df_test, predict):
    # 构建结果数据集
    result = pd.DataFrame(
        {'PassengerId': df_test['PassengerId'].as_matrix(), 'Survived': predict.astype(np.int32)})
    result.to_csv("logistic_regression_predictions.csv", index=False)
# 保存结果
saveResult(data_test, predictions)
# 优化开始
# 之前是基本模型，逻辑回归系统优化开始

# 这些系数为正的特征，和最后结果是一个正相关，反之为负相关。
# print(pd.DataFrame({"columns": list(data_train.columns)[1:], "coef": list(clf.coef_.T)}))


# 交叉验证
# 通常情况下，这么做cross validation：把train.csv分成两部分，一部分用于训练我们需要的模型，另外一部分数据上看我们预测算法的效果。
# def cv_source(df):
#     clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
#     all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
#     X = all_data.as_matrix()[:, 1:]
#     y = all_data.as_matrix()[:, 0]
#     print((clf, X, y, cv=5))
# cv_source(data_train)


def cv_source_m(df):
    for classifier in test_classifiers:
        # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

        all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        X = all_data.as_matrix()[:, 1:]
        y = all_data.as_matrix()[:, 0]
        clf = classifiers[classifier](X, y)
        cvs = model_selection.cross_val_score(clf, X, y, cv=10)
        # print(type(cvs))
        # print(classifier + ":" + str(cvs) + " mean:"+str(np.mean(cvs)))
        # msg = "%s %s " % classifier, cvs.mean()
        print("{0}  mean:{1} std: {2}".format(classifier, cvs.mean(), cvs.std()))
        # , cvs.std()
cv_source_m(data_train)


# 查看bad case
def bad_case(df):
    # 分割数据，按照 训练数据:cv数据 = 7:3的比例
    split_train, split_cv = model_selection.train_test_split(df, test_size=0.3, random_state=0)
    train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # 生成模型
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])

    # 对cross validation数据进行预测
    cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(cv_df.as_matrix()[:, 1:])
    # print(predictions)
    origin_data_train = pd.read_csv("train.csv")
    # cv_df['PassengerId'] = origin_data_train['PassengerId']
    # split_cv['PassengerId'] = origin_data_train['PassengerId']
    # 由于元数据已经切除PassengerId，所以需要重新添加
    # split_cv.loc[:, 'PassengerId'] = origin_data_train.loc[:, 'PassengerId']
    split_cv['PassengerId'] = origin_data_train['PassengerId']
    # split_cv.loc[origin_data_train['PassengerId']]
    # print(split_cv)
    # split_cv['PassengerId'] = origin_data_train['PassengerId']
    # print(split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)
    # print(split_cv)
    # print(type(origin_data_train))
    bad_cases = origin_data_train.loc[
        origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
    return bad_cases


bc = bad_case(data_train)


# print(bc)

# 学习曲线(learning curves)
# 有一个很可能发生的问题是，我们不断地做feature engineering，产生的特征越来越多.
# 用这些特征去训练模型，会对我们的训练集拟合得越来越好，同时也可能在逐步丧失泛化能力.
# 从而在待预测的数据上，表现不佳，也就是发生过拟合问题。


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def plot_validation_curve(estimator, title, X, y,):
    # 选取合适的参数gamma
    # 加载数据集

    # 定义gamma参数
    param_range = np.logspace(-1, 1)

    # 用SVM进行学习并记录loss
    train_loss, test_loss = model_selection.validation_curve(
        SVC(),
        X,
        y,
        param_name='gamma',
        param_range=param_range)

    # 训练误差均值
    train_loss_mean = -np.mean(train_loss, axis=1)
    # 训练误差标准差
    train_loss_std = np.std(train_loss, axis=1)
    # 测试误差均值
    test_loss_mean = -np.mean(test_loss, axis=1)
    # 测试误差标准差
    test_loss_std = np.std(test_loss, axis=1)


    plt.figure()
    plt.title(title)

    # 绘制误差曲线
    plt.plot(param_range, train_loss_mean, 'o-', color='r', label='Training')
    plt.plot(param_range, test_loss_mean, 'o-', color='g', label='Cross-Validation')

    plt.xlabel('Fare_scaled')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()


# 模型融合(model ensemble)
def bagging(df, df_test):
    from sklearn.ensemble import BaggingRegressor

    train_df = df.filter(
        regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到BaggingRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                                   bootstrap_features=False, n_jobs=-1)
    bagging_clf.fit(X, y)

    test = df_test.filter(
        regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
    predictions = bagging_clf.predict(test)
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv("logistic_regression_bagging_predictions.csv", index=False)


bagging(data_train, data_test)

# plot_learning_curve(clf, u"学习曲线", X, y)
# plot_validation_curve(clf, u"学习曲线", X, y)

