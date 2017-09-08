# 计算每场人数
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
path = "../zhunian_detailed.txt"
pathPd = "../test.txt"
file = open(path, encoding="utf-8")

# df['时间'] = pd.to_datetime(df['时间'],format='%Y.%m.%d')

start = time.time()
df = pd.read_csv(pathPd, delimiter=' ', header=None, names=['时间', '字母'])
# 2号
# s = pd.Series(df['字母'].str.split(',').apply(pd.Series, 1).stack())
s = df['字母'].str.split(',').apply(pd.Series, 1).stack()
# print(s)
# print(type(s))

s.index = s.index.droplevel(-1)
# print(s)

s.name = '字母'
del df['字母']
df = df.join(s)
# print(df)
df = df.drop_duplicates()
# print(df)
# print(df)
print(df.groupby('时间').size())
end = time.time()
print(end - start)

# df['字母'] = df['字母'].str.split(',').str.get()
# print(df)
print("----------------------------")
start = time.time()
df1 = pd.read_csv(pathPd, delimiter=' ', header=None, names=['时间', '字母'])
df1['字母'] = df1['字母'].str.split(',')
# df1 = pd.DataFrame({'时间':[2016,2016,2017,2017],'字母':[list('abc'),list('ab'),list('bc'),list('ab'),]})


def l_ex(*args):
    print(*args)
    l = [x for y in args for x in y]
    l = [x for y in l for x in y]
    l = set(l)
    # l = list(set(l))
    # l = len(list(set(l)))
    return l

df_m = df1.groupby('时间')['字母'].agg(l_ex)
print("----------------------------")

# print(df1)
print(df_m)
end = time.time()
print(end - start)

# s = pd.Series(df['字母'].values,index=df['时间'])
# print(s.to_period('A').index)
# s.index = s.to_period('A').index
# print(s)
# s = s.str.split(",")
# print(s)
# df1 = s.str.split(",").str.get()
# print(df1)
# df = df.str.split('，').str.get()

# data1=pd.Series.str.split(df['字母'],pat=',', expand=True)

# x = np.linspace(0, 2 * np.pi, 50)
# plt.plot(x, np.sin(x)) # 如果没有第一个参数 x，图形的 x 坐标默认为数组的索引
# plt.show() # 显示图形
# # grade_split = pd.DataFrame((x.split(',') for x in df['字母']),index=df['时间'],columns=['grade','sub_grade'])
#
# x = np.random.rand(1000)
# y = np.random.rand(1000)
# size = np.random.rand(1000) * 50
# colour = np.random.rand(1000)
# plt.scatter(x, y, size, colour)
# plt.colorbar()
# plt.show()
