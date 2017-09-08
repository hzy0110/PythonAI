import kNN

group, labels = kNN.createDataSet()
print(group)
print(labels)

print(kNN.classify0([0,0], group, labels, 3))
