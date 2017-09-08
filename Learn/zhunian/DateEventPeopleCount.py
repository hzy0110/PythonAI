import pandas as pd
import numpy as np
import time
path = "../zhunian_detailed.txt"
pathPd = "../zhunian_detailed_pd.txt"
file = open(path, encoding="utf-8")
yearPersonTime = {}
yearPerson = set()
yearPersonDict = {}
start = time.time()
for line in file.readlines():
    line = line.strip("\n")
    # 排除没有人的行
    if len(line.split("-")[1]) > 0:
        # X年有X人参加，获取年份-人名的组合，使用set去重
        year = line[8:12]
        for per in line.split("-")[1].split(","):
            yearPerson.add(year + "-" + per)

        # X年有X人次参加
        if line[8:12] in yearPersonTime:
            l = len(line.split("-")[1].split(","))
            # print(line.split("-")[1].split(","))
            # print("l1="+str(l))
            yearPersonTime[line[8:12]] += len(line.split("-")[1].split(","))
        else:
            yearPersonTime[line[8:12]] = len(line.split("-")[1].split(","))
print(sorted(yearPersonTime.items(), key=lambda x: -x[1]))

# print(yearPer)
# 去重后拆分年份和人，用年份作为key，统计人数
for p in yearPerson:
    s = p.split("-")
    if str(s[0]) in yearPersonDict:
        yearPersonDict[s[0]] += 1
    else:
        yearPersonDict[s[0]] = 1
print(sorted(yearPersonDict.items(), key=lambda x: -x[1]))
end = time.time()
print(end - start)

# X年有X人次参加
df = pd.read_csv(pathPd, delimiter='-', nrows=2, header=None, names=['编号', '启动日期', '实际第几日', '时段', '人员'])
# 转换成日期类型
df['启动日期'] = pd.to_datetime(df['启动日期'], format="%Y年%m月%d日")
# 设置索引列
df = df.set_index('启动日期')
dfYearPersonTime = pd.DataFrame(columns=['人数'])

# 统计df人员的人次数，同时会把df的index带过来
dfYearPersonTime['人数'] = df['人员'].str.count(",")+1
#把索引列只留下年份
dfYearPersonTime.index = dfYearPersonTime.index.to_period('A')
# print(len(df))
print(dfYearPersonTime.groupby(dfYearPersonTime.index).sum().sort_values(by='人数', ascending=False))
print("------------------------------------")


# X年有X人参加
start = time.time()
dfYearPerson = pd.DataFrame(columns=['人员'])
dfYearPerson['人员'] = df['人员'].str.split(",")
dfYearPerson.index = dfYearPerson.index.to_period('A')
dfYearPerson = dfYearPerson.dropna()
# print(dfYearPerson.isnull().values.any())
# print(dfYearPerson)
# print(dfYearPerson.dropna())
# print(dfYearPerson)
# print(dfYearPerson)


def l_ex(*args):
    a = [x for y in args for x in y]
    a = [x for y in a for x in y]
    # a = set(a)
    # a = list(set(a))
    a = len(list(set(a)))
    return a


def l_ex1(*args):
    a = [x for y in args for x in y]
    print("a1:")
    print(a)
    b = list()
    for y in args:
        print("y1:")
        print(y)
        for x in y:
            print("x1:")
            print(x)
            b.append(x)
    print("b1:")
    print(b)
    b.clear()

    for y in a:
        print("y2:")
        print(y)
        for x in y:
            print("x2:")
            print(x)
            b.append(x)
    print("b2:")
    print(b)
    a = [x for y in a for x in y]
    print("a2:")
    print(a)

            # a = set(a)
            # a = list(set(a))
            # a = len(list(set(a)))
    return 1

df_m = dfYearPerson.groupby(dfYearPerson.index)['人员'].agg(l_ex)
print("X年有X人参加")
print(df_m)
print("------------------------------------")
# df_m1 = dfYearPerson.groupby(dfYearPerson.index)['人员'].agg(l_ex1)
print("------------------------------------")
# print(df_m1)
# print(dfYearPerson)
end = time.time()
print(end - start)
