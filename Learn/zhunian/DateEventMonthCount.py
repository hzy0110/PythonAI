# X月产生X场
from collections import Counter
import pandas as pd
path = "../zhunian_detailed.txt"
pathPd = "../zhunian_detailed_pd.txt"
file = open(path, encoding='utf-8')
session = list()
sessionMonth = {}
# 获取场次对月份
for line in file.readlines():
    line = line.strip('\n')
    session.append(line.split(":")[0])


# 获取场次的月份
def getMonth(x):
    return x[13:15]

#最内部去重，然后提取月份，然后计算月份出现次数，然后排序
session = Counter(map(getMonth, set(session)))
print(session)

df = pd.read_csv(pathPd, delimiter='-', header=None, names=['编号', '启动日期', '实际第几日', '时段', '人员'])
# print(df)
# 把列转成日期格式
df['启动日期'] = pd.to_datetime(df['启动日期'], format='%Y年%m月%d日')
# s获取df的列，值用编号是因为一天可能会有两场
s = pd.Series(df['编号'].values, index=df['启动日期'])
# 去重
s = s.drop_duplicates()
# print(s.to_period('M'))
# print(s.index.month)
# print(type(s.index.month))
# 分组并排序,降序
print(s.groupby(s.index.month).size().sort_values(ascending=False))

# # 去重
# session = list(set(session))
#
# # print(session)
# # 调用函数，吧整个场次只留下月份
# session = list(map(f,session))
# # print(session)
#
# # 统计每个月的数量
# m=collections.Counter(session)
# print(m)

# 统计每个月的数量
# for m in session:
#     if session.count(m) > 0:
#         sessionMonth[m] = session.count(m)
# # print(sessionMonth)
# # 排序,正序
# print(sorted(sessionMonth.items(),key=lambda d:d[1]))
# # 排序，降序
# print(sorted(sessionMonth.items(),key=lambda d:-d[1]))
