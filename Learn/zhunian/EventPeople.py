# 计算每场人数
import pandas as pd
path = "../zhunian_detailed.txt"
pathPd = "../zhunian_detailed_pd.txt"
file = open(path, encoding="utf-8")

sessionAvg = {}
sessionPersonDict = {}
for line in file.readlines():
    line = line.strip("\n")
    # 排除没有人的行
    if len(line.split("-")[1]) > 0:
        # X场有X人次参加
        if line.split(":")[0] in sessionPersonDict:
            l = len(line.split("-")[1].split(","))
            # print(line.split("-")[1].split(","))
            # print("l1="+str(l))
            sessionPersonDict[line.split(":")[0]] += len(line.split("-")[1].split(","))
        else:
            sessionPersonDict[line.split(":")[0]] = len(line.split("-")[1].split(","))


print(sorted(sessionPersonDict.items(), key=lambda x: -x[1]))

df = pd.read_csv(pathPd, delimiter='-', header=None, names=['编号', '启动日期', '实际第几日', '时段', '人员'])
# 定义一个新的df，设置列名
dfp = pd.DataFrame(columns=['启动日期', '人员数'])

# 把2个列拼接到1个列
dfp['启动日期'] = df['编号'] + "-" + df['启动日期']
# 统计人员里逗号出现的次数+1，获得人数
dfp['人员数'] = df['人员'].str.count(",")+1
print(dfp)
print("---------------")
# 根据场次groupby，加和数值列，然后根据人数列降序
print(dfp.groupby(dfp['启动日期']).sum().sort_values(by='人员数', ascending=False))
# print(dfp.dtypes)
