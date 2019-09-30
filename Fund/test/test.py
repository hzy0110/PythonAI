import pandas as pd
import arrow
from tools.tool import Tool
#


# a = arrow.now()
# print(a)
# b = arrow.get('2010-01-15')
# print(b)
#
#
# d = arrow.get('2005-10-29')
# c = b - d
# print(c, type(c), c.days)
#
#
# def getTimedelta(dayStr):
#     if dayStr == '无日期':
#         return 0
#     now = arrow.now()
#     target_day = arrow.get(dayStr)
#     diff = now - target_day
#     return diff.days
# print(arrow.now().date())
a = Tool().getTradeDayList('')
print(a)
