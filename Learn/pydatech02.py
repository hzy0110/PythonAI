#coding=utf-8
import json
from collections import defaultdict
from collections import Counter
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
path = "usagov_bitly_data2012-03-16-1331923249.txt"
open(path).readline()
records = [json.loads(line) for line in open(path)]
records[0]
records[0]['tz']
print(records[0]['tz'])
#timezone = [rec['tz'] for rec in records]
timezone = [rec['tz'] for rec in records if 'tz' in rec]
timezone[:10]

# print timezone
# 分别统计各个时区有多少
def getCounts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

def getCounts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts


counts = getCounts(timezone)
print(counts)
print(counts['America/New_York']) #计算America/New_York有几个

print(len(timezone))#行数

def topCounts(countDict,n =10):
    valueKeyPairs = [(count, tz) for  tz, count in countDict.items()]
    valueKeyPairs.sort()
    return valueKeyPairs[-n:]
print('---------------topCounts----------------')

print(topCounts(counts))

#使用Counter函数，简化方法
counts = Counter(timezone)
print('---------------most_common----------------')
print(counts.most_common(10))

frame = DataFrame(records)
print('---------------frame----------------')
#print(frame)

print(frame['tz'][:10])


tzCounts = frame['tz'].value_counts()
print(tzCounts[:10])

cleanTz = frame['tz'].fillna('Missing')
cleanTz[cleanTz == ''] = 'Unknown'

tzCounts = cleanTz.value_counts()
print(tzCounts[:10])

plt.rc('figure', figsize=(10, 6))
plt.figure(figsize=(10, 4))

tzCounts[:10].plot(kind='barh', rot=0)
#加上这个才能显示图像
#plt.show()

#打印a字段对第x行
print(frame['a'][0])
print(frame['a'][50])

#获取a字段对数据
results = Series([x.split()[0] for x in frame.a.dropna()])
#打印前五
print(results[:5])

#统计results里，总数前10的
print(results.value_counts()[:10])

cframe = frame[frame.a.notnull()]

#根据a是否包含windows来判断是否windows
operatingSystem = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
print(operatingSystem)
byTzOs = cframe.groupby(['tz', operatingSystem])

aggCounts = byTzOs.size().unstack().fillna(0)
print(aggCounts[:10])

indexer = aggCounts.sum(1).argsort()
print(indexer[:10])

countSubset = aggCounts.take(indexer)[-10:]
print(countSubset)

countSubset.plot(kind='barh', stacked=True)
plt.show()

plt.figure()
normedSubset = countSubset.div(countSubset.sum(1), axis=0)
normedSubset.plot(kind='barh', stacked=True)
plt.show()