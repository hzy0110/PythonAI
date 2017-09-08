# -*- coding: utf-8 -*-
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# print(check_output(["dir", "./"]).decode("utf8"))

# 加载数据
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# 定义若干个对数据进行清理的函数，这些函数主要作用在pandas的DataFrame数据类型上
# 查看数据集属性值得确实情况
def show_missing(houseprice):
    missing = houseprice.columns[houseprice.isnull().any()].tolist()
    return missing


# 查看 categorical 特征的值情况
def cat_exploration(houseprice, column):
    print(houseprice[column].value_counts())


# 对数据集中某一列的缺失值进行补全
def cat_imputation(houseprice, column, value):
    houseprice.loc[houseprice[column].isnull(), column] = value


# LotFrontage
# check correlation with LotArea
print(test_data['LotFrontage'].corr(test_data['LotArea']))  # 0.64
print(train_data['LotFrontage'].corr(train_data['LotArea']))  # 0.42
test_data['SqrtLotArea'] = np.sqrt(test_data['LotArea'])
train_data['SqrtLotArea'] = np.sqrt(train_data['LotArea'])

# print(test_data['LotFrontage'].corr(test_data['SqrtLotArea']))
# print(train_data['LotFrontage'].corr(train_data['SqrtLotArea']))

cond = test_data['LotFrontage'].isnull()
test_data.LotFrontage[cond] = test_data.SqrtLotArea[cond]  # 缺失值用房屋边长补全
cond = train_data['LotFrontage'].isnull()
train_data.LotFrontage[cond] = train_data.SqrtLotArea[cond]

del test_data['SqrtLotArea']
del train_data['SqrtLotArea']

# MSZoning
# 在test测试集中有缺失, train中没有
cat_exploration(test_data, 'MSZoning')
print(test_data[test_data['MSZoning'].isnull() is True])
# MSSubClass  MSZoning
print(pd.crosstab(test_data.MSSubClass, test_data.MSZoning))
# test_data中建筑类型缺失值补齐 30:RM 20:RL 70:RM
test_data.loc[test_data['MSSubClass'] == 20, 'MSZoning'] = 'RL'
test_data.loc[test_data['MSSubClass'] == 30, 'MSZoning'] = 'RM'
test_data.loc[test_data['MSSubClass'] == 70, 'MSZoning'] = 'RM'

# Alley
print(cat_exploration(test_data, 'Alley'))
print(cat_exploration(train_data, 'Alley'))
# Alley这个特征有太多的nans,这里填充None，也可以直接删除，不使用。后面在根据特征的重要性选择特征是，也可以舍去
cat_imputation(test_data, 'Alley', 'None')
cat_imputation(train_data, 'Alley', 'None')

# Utilities
# 只有test有缺失值, train中没有
# 并且这个column中值得分布极为不均匀
# drop 将该列属性删去
print(cat_exploration(test_data, 'Utilities'))
print(cat_exploration(train_data, 'Utilities'))
print(test_data.loc[test_data['Utilities'].isnull() == True])
test_data = test_data.drop(['Utilities'], axis=1)
train_data = train_data.drop(['Utilities'], axis=1)

# Exterior1st & Exterior2nd
# 只在test中出现缺失值(nans only appear in test set)
# 检查Exterior1st 和 Exterior2nd 是否存在缺失值共现的情况
cat_exploration(test_data, 'Exterior1st')
cat_exploration(train_data, 'Exterior1st')
print(test_data[['Exterior1st', 'Exterior2nd']][test_data['Exterior1st'].isnull() == True])
print(pd.crosstab(test_data.Exterior1st, test_data.ExterQual))
test_data.loc[test_data['Exterior1st'].isnull(), 'Exterior1st'] = 'VinylSd'
test_data.loc[test_data['Exterior2nd'].isnull(), 'Exterior2nd'] = 'VinylSd'

# MasVnrType & MasVnrArea
print(test_data[['MasVnrType', 'MasVnrArea']][test_data['MasVnrType'].isnull() == True])
print(train_data[['MasVnrType', 'MasVnrArea']][train_data['MasVnrType'].isnull() == True])
# So the missing values for the "MasVnr..." Variables are in the same rows.
cat_exploration(test_data, 'MasVnrType')
cat_exploration(train_data, 'MasVnrType')
cat_imputation(test_data, 'MasVnrType', 'None')
cat_imputation(train_data, 'MasVnrType', 'None')
cat_imputation(test_data, 'MasVnrArea', 0.0)
cat_imputation(train_data, 'MasVnrArea', 0.0)

# basement
# train
basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2']
print(train_data[basement_cols][train_data['BsmtQual'].isnull() == True])
for cols in basement_cols:
    if 'FinFS' not in cols:  # 判断字段中是否包含'FinFS'
        cat_imputation(train_data, cols, 'None')

# test
basement_cols = ['Id', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
print(test_data[basement_cols][test_data['BsmtCond'].isnull() == True])
# 其中,有三行只有BsmtCond为NaN,该三行的其他列均有值  580  725  1064
print(pd.crosstab(test_data.BsmtCond, test_data.BsmtQual))
"""
BsmtQual   Ex  Fa   Gd   TA
BsmtCond
Fa          0  14    7   37
Gd         12   2   30   13
Po          1   1    0    1
TA        124  36  553  581
"""
test_data.loc[test_data['Id'] == 580, 'BsmtCond'] = 'TA'
test_data.loc[test_data['Id'] == 725, 'BsmtCond'] = 'TA'
test_data.loc[test_data['Id'] == 1064, 'BsmtCond'] = 'TA'
# 除了上述三行之外, 其他行的NaN都是一样的
'''
for cols in basement_cols:
    if 'SF' not in cols  and 'Bath' not in cols :
        test_data.loc[test_data['BsmtFinSF1'] == 0.0, cols] = 'None'
'''
for cols in basement_cols:
    if cols not in 'SF' and cols not in 'Bath':
        test_data.loc[test_data['BsmtFinSF1'] == 0.0, cols] = 'None'
for cols in basement_cols:
    if test_data[cols].dtype == np.object:
        cat_imputation(test_data, cols, 'None')
    else:
        cat_imputation(test_data, cols, 0.0)
cat_imputation(test_data, 'BsmtFinSF1', '0')
cat_imputation(test_data, 'BsmtFinSF2', '0')
cat_imputation(test_data, 'BsmtUnfSF', '0')
cat_imputation(test_data, 'TotalBsmtSF', '0')
cat_imputation(test_data, 'BsmtFullBath', '0')
cat_imputation(test_data, 'BsmtHalfBath', '0')
'''
对于BsmtQual这个特征，取值有 Ex，Gd，TA，Fa，Po. 从数据的说明中可以看出，
这依次是优秀，好，次好，一般，差几个等级，这具有明显的可比较性，可以使用map编码
'''
train_data = train_data.replace({'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test_data = test_data.replace({'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})

# KitchenQual
# 只在测试集中有缺失值
cat_exploration(test_data, 'KitchenQual')
cat_exploration(train_data, 'KitchenQual')
print(test_data[['KitchenQual', 'KitchenAbvGr']][test_data['KitchenQual'].isnull() == True])
print(pd.crosstab(test_data.KitchenQual, test_data.KitchenAbvGr))
cat_imputation(test_data, 'KitchenQual', 'TA')

# Functional
# 只在测试集中有缺失值
# 填充一个最常见的值
cat_exploration(test_data, 'Functional')
cat_imputation(test_data, 'Functional', 'Typ')

# FireplaceQu & Fireplaces
cat_exploration(test_data, 'FireplaceQu')
cat_exploration(train_data, 'FireplaceQu')
print(test_data['Fireplaces'][test_data['FireplaceQu'].isnull() == True].describe())
print(train_data['Fireplaces'][train_data['FireplaceQu'].isnull() == True].describe())
cat_imputation(test_data, 'FireplaceQu', 'None')
cat_imputation(train_data, 'FireplaceQu', 'None')

# Garage 车库
# train set
garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
print(train_data[garage_cols][train_data['GarageType'].isnull() == True])
for cols in garage_cols:
    if train_data[cols].dtype == np.object:
        cat_imputation(train_data, cols, 'None')
    else:
        cat_imputation(train_data, cols, 0)

# test set
garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
# print(test_data[garage_cols][test_data['GarageType'].isnull() == True])
for cols in garage_cols:
    if test_data[cols].dtype == np.object:
        cat_imputation(test_data, cols, 'None')
    else:
        cat_imputation(test_data, cols, 0)

# PoolQC
# 不易处理, 并且分布偏差大, drop
print(cat_exploration(test_data, 'PoolQC'))
print(cat_exploration(train_data, 'PoolQC'))
print(test_data.loc[test_data['PoolQC'].isnull() == True])
test_data = test_data.drop(['PoolQC'], axis=1)
train_data = train_data.drop(['PoolQC'], axis=1)
test_data = test_data.drop(['PoolArea'], axis=1)
train_data = train_data.drop(['PoolArea'], axis=1)

# Fence
cat_exploration(test_data, 'Fence')
cat_exploration(train_data, 'Fence')
cat_imputation(test_data, 'Fence', 'None')
cat_imputation(train_data, 'Fence', 'None')

# MiscFeature
cat_exploration(test_data, 'MiscFeature')
cat_exploration(train_data, 'MiscFeature')
cat_imputation(test_data, 'MiscFeature', 'None')
cat_imputation(train_data, 'MiscFeature', 'None')

# SaleType
# nans only appear in test set
cat_exploration(test_data, 'SaleType')
cat_imputation(test_data, 'SaleType', 'WD')

# Electrical
# nans only appear in train set
cat_exploration(train_data, 'Electrical')
cat_imputation(train_data, 'Electrical', 'SBrkr')
'''
到此为止，我们基本把所有的缺失值都填补完整了，
但是还有一列MSSubClass，原始数据类型是int64,我并不认为这一列具有可比性，所以把MSSubClass映射成object
'''

# convert MSSubClass to object
cat_exploration(train_data, 'MSSubClass')
train_data = train_data.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E",
                                                60: "F", 70: "G", 75: "H", 80: "I", 85: "J",
                                                90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})
test_data = test_data.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E",
                                              60: "F", 70: "G", 75: "H", 80: "I", 85: "J",
                                              90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})
'''
之后，将所有categorical类型的特征进行one-hot编码。需要注意的是：训练集和测试集中，相同的列可能会有不同的类型需要统一：
'''
for col in test_data.columns:
    t1 = test_data[col].dtype
    t2 = train_data[col].dtype
    if t1 != t2:
        print(col, t1, t2)

# convert to type of int64
c = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for cols in c:
    # tmp_col = test_data[cols].astype(pd.np.float64)#应该是int64吧
    tmp_col = test_data[cols].astype(pd.np.int64)
    tmp_col = pd.DataFrame({cols: tmp_col})
    del test_data[cols]
    test_data = pd.concat((test_data, tmp_col), axis=1)

# 进行one-hot编码
for cols in train_data.columns:
    if train_data[cols].dtype == np.object:
        train_data = pd.concat((train_data, pd.get_dummies(train_data[cols], prefix=cols)), axis=1)
        del train_data[cols]

for cols in test_data.columns:
    if test_data[cols].dtype == np.object:
        test_data = pd.concat((test_data, pd.get_dummies(test_data[cols], prefix=cols)), axis=1)
        del test_data[cols]
# 特征对其时会将train_set中SalePrice,'Id'删去
train_y = train_data['SalePrice']

# 特征对齐
col_train = train_data.columns
col_test = test_data.columns
for index in col_train:
    if index in col_test:
        pass
    else:
        del train_data[index]

col_train = train_data.columns
col_test = test_data.columns
for index in col_test:
    if index in col_train:
        pass
    else:
        del test_data[index]

"""
RF特征重要性选择
"""
print('Id' in train_data.columns)
print('Id' in test_data.columns)
from sklearn.ensemble import RandomForestRegressor

etr = RandomForestRegressor(n_estimators=400)
train_x = train_data
# train_x = train_data.drop(['SalePrice', 'Id'], axis=1)
etr.fit(train_x, train_y)
print(etr.feature_importances_)
imp = etr.feature_importances_
imp = pd.DataFrame({'feature': train_x.columns, 'score': imp})
print(imp.sort(['score'], ascending=[0]))  # 按照特征重要性, 进行降序排列, 最重要的特征在最前面
imp = imp.sort(['score'], ascending=[0])
imp.to_csv("feature_importances2.csv", index=False)

# bivariate analysis saleprice/grlivarea
imp[['feature', 'score']].plot(kind='bar', stacked=True)
# 筛选前20个特征
select_feature = imp['feature'][:20]
xtrain_feature = train_x.loc[:, select_feature]

rf_select_model = etr.fit(xtrain_feature, train_y)

# write submission
subm = pd.read_csv("sample_submission.csv")
xtest_feature = test_data.loc[:, select_feature]
subm.iloc[:, 1] = np.array(rf_select_model.predict(np.array(xtest_feature)))
cat_exploration(subm, 'SalePrice')
# subm['SalePrice'] = np.expm1(subm[['SalePrice']])
subm.to_csv('RandomForest-submission1.csv', index=None)
