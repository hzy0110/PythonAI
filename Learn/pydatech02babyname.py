from __future__ import division
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 5))
np.set_printoptions(precision=4)


import pandas as pd
#指定3个列的列名
names1880 = pd.read_csv('names/yob1880.txt', names=['name', 'sex', 'births'])
print(names1880)

#打印根据性别分组的数量
print(names1880.groupby('sex').births.sum())

#添加一个数字列表
years = range(1880, 2011)

#去读所有的年份文件，并且增加一列年份
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = 'names/yob%d.txt' %year
    frame = pd.read_csv(path, names=columns)
    frame['year']=year
    pieces.append(frame)
#将所有数据整合到单个DateFrame中
names = pd.concat(pieces, ignore_index=True)
#计算每年人口出生根据性别
totalBirths = names.pivot_table('births', index='year', columns='sex', aggfunc=sum)
print(totalBirths.tail())

#totalBirths.plot(title='Total births by sex and year')


#计算每个名字在总出生的占比
def addProp(group):
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group

names = names.groupby(['year', 'sex']).apply(addProp)
print(names)

#做有效性检查，检查看每个分组的prop的总和是否是1
print(np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1))

#计算每对sex／year组合的前1000个名字
def getTop1000(group):
    return group.sort_values(by='births', ascending=False)[:1000]



grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(getTop1000)

print(top1000)

boys = top1000[top1000.sex == "M"]
girls = top1000[top1000.sex == 'F']

#根据year和name统计总出生数透视表
totalBirths = top1000.pivot_table('births', index='year', columns='name', aggfunc=sum)

#用DataFrame的plot方法绘制几个名字的曲线图
subset = totalBirths[['John', 'Harry', 'Mary', 'Marilyn']]
#subset.plot(subplots=True, figsize=(12, 10), grid=False, title='Number of births per year')

#计算前1000人名对所有人名占比
tables = top1000.pivot_table('prop', index='year', columns='sex', aggfunc=sum)
#tables.plot(title='计算前1000人名对所有人名占比', yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))

#计算相应年前10个名字对占比
df = boys[boys.year == 1900]
prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
print(prop_cumsum[:10])
#利用加和对方法，计算到超过50%对人名，需要前几对人名,数组从零开始的，所以要对结果加一
print(prop_cumsum.values.searchsorted(0.5))

def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1


diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
diversity.head()
#名字对多样性变化
#diversity.plot(title="历年50%占比对名字数量")


#从name列取出最后一个字母
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'

table = names.pivot_table('births', index=last_letters, columns=['sex', 'year'], aggfunc=sum)


subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
print(subtable.head())

#计算各性别各末尾字母对出生率
letter_prop = subtable / subtable.sum().astype(float)


#根据字母在各年份比例生成条形图
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
#letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
#letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)
plt.subplots_adjust(hspace=0.25)

letter_prop = table / table.sum().astype(float)

#计算已dny结尾的男孩的名字的人数比例
dny_ts = letter_prop.ix[['d', 'n', 'y'], 'M'].T
dny_ts.head()

dny_ts.plot()
#plt.close('all')


#统计男孩取女孩名及相反的情况
all_names = top1000.name.unique()
#查找含有lesl这样的名字
mask = np.array(['le·sl' in x.lower() for  x in all_names])
lesley_like = all_names[mask]
print(lesley_like)

#查看名字分组和对应频率
filtered = top1000[top1000.name.isin(lesley_like)]
print(filtered.groupby('name').births.sum())


#按照性别和年度聚合，按年度进行规范化处理
table = filtered.pivot_table('births', index='year', columns='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
print(table.tail())

table.plot(style={'M': 'k-', 'F': 'k--'})

plt.show()