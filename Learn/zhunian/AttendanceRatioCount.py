import numpy as np
import pandas as pd
import collections
import codecs
from collections import Counter
import time
pathPd = "../zhunian_detailed_pd.txt"
# print("你好，世界")
path = "../zhunian_detailed.txt"
path1 = "../test.txt"
file = open(path, encoding='utf-8')
session = list()
personnel = list()
person = list()
start = time.time()
# 每个人对每场对出勤率
# 按行获取，冒号前的统计场次，然后取出所有人数
for line in file.readlines():
    line = line.strip('\n')
    session.append(line.split(":")[0])
    if len(line.split("-")[1]) > 0:
        personnel.append(line.split(":")[0]+":"+line.split("-")[1])


# 把每个场次加到人名上
def f(x):
    # print("x:"+x)
    p = x.split(":")[1].split(",")
    # print("p:")
    # print(p)
    p1 = list()
    for s in p:
        # print("s:"+s)
        # print("x1:"+x.split(":")[0]+ "-" + s)
        p1.append(x.split(":")[0] + "-" + s)
    return p1


def f1(x):
    return x.split("-")[1]

attPerList = list(map(f, personnel))
print("attPerList:"+str(len(attPerList)))

#去重场次数
sessionTotal = len(set(session))

print("sessionTotalLen:"+str(sessionTotal))
print("personnelLen:"+str(len(personnel)))


#分割每行的人
# print("11111111111")
for i in attPerList:
    # print("i:"+str(i))
    for j in i:
        # print("j:" + str(j))
        person.append(j)
# 去重场次和人名
print("personLen:"+str(len(person)))
person = list(set(person))
print("personLen:"+str(len(person)))
# print(person)
person = list(map(f1, person))
# print(person)

personnelTotal = {}
personnelRatio = {}
# for i in person:
#     if person.count(i) > 1:
#         personnelTotal[i] = person.count(i)
# print(personnelTotal)
# 按照不同的人，计算每人出现总次数
personnelTotal = dict(collections.Counter(person))
for k in personnelTotal:
    personnelRatio[k] = personnelTotal[k]/sessionTotal

# print(personnelRatio)
# 排序
print(sorted(personnelRatio.items(), key=lambda d: -d[1]))


end = time.time()
print(end - start)

# pd
start = time.time()
df = pd.read_csv(pathPd, delimiter='-', header=None, names=['编号', '启动日期', '实际第几日', '时段', '人员'])
# 去除空行
df = df.dropna()
df = df.set_index('编号')
df['人员'] = df['人员'].str.split(",")

sSessionTotal = pd.Series(df.index)
# print(sSessionTotal)
# 去重
sSessionTotal = sSessionTotal.drop_duplicates()
SessionTotal = len(sSessionTotal)


def l_ex(*args):
    """
    groupby时，重组元素
    :param args:
    :return:
    """
    # print(*args)
    l = [x for y in args for x in y]
    l = [x for y in l for x in y]
    # l = set(l)
    # l = list(set(l))
    l = len(list(set(l)))
    return l


def l_ex1(*args):
    """
    吧相同组的内容，合并到一个list
    :param args:
    :return:
    """
    a = [x for y in args for x in y]
    # print("a1:")
    # print(a)
    b = list()
    for y in args:
        # print("y1:")

        # print(y)
        # print("y1type:")
        # print(type(y))
        for x in y:
            # print("x1:")
            # print(x)
            b.append(x)
    # print("b1:")
    # print(b)
    b.clear()

    for y in a:
        # print("y2:")
        # print(y)
        for x in y:
            # print("x2:")
            # print(x)
            b.append(x)
    # print("b2:")
    # print(b)
    a = [x for y in a for x in y]
    # print("a2:")
    # print(a)
    # print("***")

    # a = set(a)
    a = list(set(a))
    # a = len(list(set(a)))
    return a


# df_m = df.groupby('编号')['人员'].agg(l_ex1)
# 分组，并调用方法，吧同组的人合到一起
df_m = df.groupby(df.index)['人员'].agg(l_ex1)
# print(df_m)
# print(type(df_m))
# 行转列
s = df_m.apply(pd.Series, 1).stack()
# print(s)
# print(type(s))
# 删除二级分组
s.index = s.index.droplevel(-1)
# print(s)
# s = s.values
# print(s)
# 根据人再次分组
s1 = s.groupby(s.values)
s1 = s1.count()
# 计算出勤率
s1 = s1.map(lambda x: x/SessionTotal).sort_values(ascending=False)
print(s1)
# print(type(s1.count()))
# print(df_t)
end = time.time()
print(end - start)

#出勤率，每个人在所有场次里出现的次数
#取出冒号前面的，去重，获得总共的场次数
# val eDate = textFile.map(line => line.split(":")(0)).distinct().count().toDouble
#     val pcDate = textFile.map(line => line.split("-")).
#       filter(line => line.length > 1).
# (No.001)2012年07月22日:人名
#       map(line => (line(0).split(":")(0) + ":" + line(1))).
# (No.001)2012年07月22日,人名根据逗号分割
#       map(line => (line.split(":")(0), line.split(":")(1).split(","))).
# 人名-(No.001)2012年07月22日去重
#       flatMap(line => (line._2.map(l2 => l2 + "-" + line._1))).distinct().
# 去重后保留唯一人名+场次,1
#       map(line => (line.split("-")(0), 1)).
# 加和
#       reduceByKey((a, b) => a + b).
# 处以场次数
#       map(line => (line._1, line._2.toDouble / eDate)).
# 排序
#       sortBy(_._2, false)
