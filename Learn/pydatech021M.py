import pandas as pd
import os
encoding = 'latin1'

upath = os.path.expanduser('movielens/users.dat')
rpath = os.path.expanduser('movielens/ratings.dat')
mpath = os.path.expanduser('movielens/movies.dat')

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
mnames = ['movie_id', 'title', 'genres']

users = pd.read_table(upath, sep='::', header=None, names=unames, encoding=encoding, engine='python')
ratings = pd.read_csv(rpath, sep='::', header=None, names=rnames, encoding=encoding, engine='python')
movies = pd.read_csv(mpath, sep='::', header=None, names=mnames, encoding=encoding, engine='python')
#合并电影用户评分3个表
data = pd.merge(pd.merge(ratings, users), movies)
print(data[:10])
#输出合并表第0行
print(data.ix[0])
#聚合评分表，根据标题
meanRatings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
print(meanRatings[:5])

#根据标题分组
ratingsByTitle = data.groupby('title').size()
print(ratingsByTitle[:10])

#找出评分次数250次以上的
activeTitle = ratingsByTitle.index[ratingsByTitle >= 250]
print(activeTitle)
#导入有250次评分的电影
meanRatings =meanRatings.ix[activeTitle]
print(meanRatings.ix[activeTitle])

#根据女性打分，降序
topFemaleRatings = meanRatings.sort_values(by='F', ascending=False)
print(topFemaleRatings)

#计算男女评分差异
meanRatings['diff'] = meanRatings['M'] - meanRatings['F']

#找到男女差异最大的电影
sortedByDiff = meanRatings.sort_values(by='diff')
print(sortedByDiff[:10])
print('降序取出男性喜欢的电影')
#负数的步长
print(sortedByDiff[::-1][:15])

print('计算电影的标准差')
ratingsStdBytitle = data.groupby('title')['rating'].std()

#根据activeTitles过滤出有250条评分的电影
ratingsStdBytitle = ratingsStdBytitle.ix[activeTitle]

#降序排列打印
print(ratingsStdBytitle.sort_values(ascending=False)[:10])



