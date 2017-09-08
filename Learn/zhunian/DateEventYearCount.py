# default_encoding = 'utf-8'
# import importlib
# importlib.reload(sys)
import pandas as pd
import numpy as np
# X年产生X场
from collections import Counter
import time
import matplotlib.pyplot as plt
start = time.time()
path = "../zhunian_detailed.txt"
pathPd = "../zhunian_detailed_pd.txt"
file = open(path, encoding="utf-8")
session = list()

for line in file.readlines():
    line = line.strip('\n')
    session.append(line.split(":")[0])


def getYear(x):
    return x[8:12]
print(set(session))
#最内部去重，然后提取年份，然后计算年份出现次数，然后排序
session = Counter(map(getYear, set(session)))
print(session)

end = time.time()
print(end - start)

start = time.time()
df = pd.read_csv(pathPd, delimiter="-", header=None, parse_dates=True, names=['编号', '启动日期', '实际第几日', '时段', '人员'])
# print(df)
# print(df.dtypes)
# 将数据类型转换为日期类型
df['启动日期'] = pd.to_datetime(df['启动日期'], format="%Y年%m月%d日")
# print(df.dtypes)
# print(df)
# print(df['启动日期'])
# df['date'] ='2014'.head()
# 设置索引列
# df = df.set_index('启动日期')
# print(df['2014'])

# print(df[df['启动日期'] == '2012-07'].head())
# 打印变量类型
# print(type(df))
# print(type(df.index))
# print(df['2013'].head(2))
# 获取某咧给Series,要.values才是获得值,索引使用df的索引，值用编号是因为一天可能会有两场
s = pd.Series(df['编号'].values, index=df['启动日期'])

# s = pd.Series(df['启动日期'].values, index=df.index)
# Serise转DataFrame
# df1 = pd.DataFrame(s)
# s.head(2)
# print(type(s.index))
# print(s)
# 判断是否有重复的key
# IsDuplicated = s.duplicated()
# IsDuplicated1 = df.duplicated()
# print(IsDuplicated)
# print(IsDuplicated1)
# 去重,只能根据列去重，不能根据index
print(s.index)
s = s.drop_duplicates()
# print(s['2014'])
# print(s.resample('A').sum())
# print(s.groupby(s.index).size())
# print(s.to_period('A'))
# 用根据年份显示的index列，作为group依据
s = s.groupby(s.index.to_period('A')).size()  # .sort_values(ascending=False)
print(s)
# print(df1.to_period['M'])
# print(df.resample('A').sum().to_period('A'))
# print(df.resample('w').sum().head())
end = time.time()
print(end - start)
print(s.index)
print(s.values)

# s.plot()
# plt.plot(s.values, s.index) # 如果没有第一个参数 x，图形的 x 坐标默认为数组的索引
# plt.show(s.plot(kind='bar'))


# s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))

plt.show(s.plot())
plt.show(s.plot(kind='bar'))
