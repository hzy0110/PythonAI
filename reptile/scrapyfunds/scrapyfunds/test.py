# import numpy as np
import pandas as pd
#
# fund_grant_total_str = 'var ljsylSVG_PicData="2019/03/25_0.00_0.00_0.00|2019/03/26_-1.65_-1.13_-1.51|2019/03/27_0.16_0.02_-0.67|2019/03/28_-0.79_-0.39_-1.58|2019/03/29_2.13_3.46_1.57|2019/04/01_3.31_6.17_4.18|2019/04/02_1.73_6.10_4.40|2019/04/03_1.65_7.46_5.69|2019/04/04_2.44_8.53_6.69|2019/04/08_1.02_8.40_6.63|2019/04/09_2.44_8.89_6.46|2019/04/10_3.23_9.16_6.54|2019/04/11_-0.08_6.81_4.83|2019/04/12_0.39_6.57_4.78|2019/04/15_-0.79_6.22_4.43|2019/04/16_1.18_9.16_6.92|2019/04/17_2.21_9.20_7.23|2019/04/18_1.73_8.80_6.81|2019/04/19_2.36_10.09_7.48|2019/04/22_2.36_7.56_5.65|2019/04/23_1.65_7.38_5.11|2019/04/24_1.81_7.68_5.21|2019/04/25_0.55_5.32_2.66|2019/04/26_0.08_3.91_1.43|2019/04/29_1.34_4.21_0.64|2019/04/30_1.18_4.55_1.16|2019/05/06_-3.39_-1.56_-4.49|2019/05/07_-1.10_-0.59_-3.83|2019/05/08_-3.15_-2.01_-4.91|2019/05/09_-4.73_-3.82_-6.31|2019/05/10_-1.34_-0.33_-3.41|2019/05/13_-1.97_-1.98_-4.58|2019/05/14_-2.21_-2.61_-5.24|2019/05/15_1.42_-0.42_-3.43|2019/05/16_0.95_0.03_-2.87|2019/05/17_-1.73_-2.51_-5.28|2019/05/20_-3.78_-3.34_-5.67|2019/05/21_-2.92_-2.03_-4.50|2019/05/22_-2.60_-2.50_-4.97|2019/05/23_-4.57_-4.24_-6.26|2019/05/24_-4.41_-3.98_-6.24|2019/05/27_-3.07_-2.82_-4.95|2019/05/28_-2.21_-1.89_-4.37|2019/05/29_-2.76_-2.11_-4.22|2019/05/30_-2.76_-2.72_-4.51|2019/05/31_-2.68_-3.02_-4.74|2019/06/03_-3.55_-2.96_-5.03|2019/06/04_-4.96_-3.86_-5.94|2019/06/05_-6.30_-3.89_-5.97|2019/06/06_-7.80_-4.76_-7.07|2019/06/10_-6.15_-3.53_-6.27|2019/06/11_-2.84_-0.63_-3.86|2019/06/12_-3.78_-1.38_-4.39|2019/06/13_-3.47_-1.53_-4.35|2019/06/14_-4.10_-2.35_-5.29|2019/06/17_-4.96_-2.35_-5.11|2019/06/18_-4.33_-2.01_-5.02|2019/06/19_-2.60_-0.72_-4.12|2019/06/20_0.08_2.29_-1.84|2019/06/21_0.87_2.43_-1.35|2019/06/24_1.34_2.63_-1.15|2019/06/25_0.71_1.56_-2.00|2019/06/26_2.52_1.38_-2.19|2019/06/27_4.18_2.46_-1.52|2019/06/28_3.86_2.21_-2.11|2019/07/01_6.78_5.16_0.06|2019/07/02_7.80_5.19_0.03|2019/07/03_5.59_4.03_-0.91|2019/07/04_4.18_3.48_-1.24|2019/07/05_5.59_4.02_-1.05|2019/07/08_3.31_1.60_-3.60|2019/07/09_3.31_1.34_-3.77|2019/07/10_3.47_1.17_-4.20|2019/07/11_2.92_1.13_-4.12|2019/07/12_3.62_1.76_-3.70|2019/07/15_3.78_2.17_-3.31|2019/07/16_3.15_1.71_-3.46|2019/07/17_2.68_1.65_-3.66|2019/07/18_2.29_0.68_-4.66|2019/07/19_2.76_1.74_-3.91|2019/07/22_3.70_1.04_-5.13|2019/07/23_4.18_1.26_-4.70|2019/07/24_4.96_2.06_-3.94|2019/07/25_5.12_2.89_-3.47|2019/07/26_6.15_3.09_-3.24|2019/07/29_6.15_2.98_-3.35|2019/07/30_6.78_3.41_-2.98|2019/07/31_6.15_2.47_-3.63|2019/08/01_5.75_1.62_-4.41|2019/08/02_5.83_0.12_-5.76|2019/08/05_4.33_-1.79_-7.28|2019/08/06_4.73_-2.85_-8.72|2019/08/07_4.89_-3.24_-9.02|2019/08/08_7.41_-1.96_-8.17|2019/08/09_6.78_-2.92_-8.82|2019/08/12_8.59_-1.17_-7.49|2019/08/13_9.14_-2.06_-8.08|2019/08/14_11.27_-1.61_-7.69|2019/08/15_11.58_-1.30_-7.47|2019/08/16_12.45_-0.86_-7.20|2019/08/19_14.26_1.29_-5.26|2019/08/20_14.18_1.20_-5.36|2019/08/21_14.89_1.04_-5.35|2019/08/22_15.29_1.35_-5.24|2019/08/23_18.36_2.09_-4.78|2019/08/26_17.26_0.62_-5.90|2019/08/27_18.36_1.98_-4.63|2019/08/28_18.44_1.60_-4.91|2019/08/29_18.68_1.27_-5.00|2019/08/30_20.09_1.52_-5.15|2019/09/02_22.06_2.82_-3.91|2019/09/03_22.14_2.96_-3.71|2019/09/04_20.80_3.83_-2.81|2019/09/05_20.41_4.88_-1.88|2019/09/06_20.02_5.50_-1.43|2019/09/09_20.33_6.15_-0.60|2019/09/10_20.02_5.78_-0.72|2019/09/11_17.10_5.00_-1.12|2019/09/12_18.20_6.13_-0.39|2019/09/16_19.15_5.74_-0.40|2019/09/17_18.28_3.96_-2.13|2019/09/18_20.72_4.47_-1.89|2019/09/19_21.12_4.85_-1.44|2019/09/20_23.25_5.15_-1.20|2019/09/23_22.62_3.95_-2.17|2019/09/24_26.32_4.23_-1.90";'
#
# fund_grant_total_str = fund_grant_total_str.replace('var ljsylSVG_PicData=', '')
# fund_grant_total_str = fund_grant_total_str.replace('"', '')
#
# fund_grant_total_list1 = fund_grant_total_str.split('|')
# # print(len(t1_list))
# fund_grant_total_list2 = []
# for s in fund_grant_total_list1:
#     fund_grant_total_list2.append([s.split('_')])
# # print(len(t2_list))
# fund_grant_total_np1 = np.array(fund_grant_total_list2)
# print(fund_grant_total_np1.shape)
# fund_grant_total_np2 = fund_grant_total_np1.reshape(fund_grant_total_np1.shape[0], fund_grant_total_np1.shape[2])
# print(fund_grant_total_np2.shape)
#
# # fund_grant_total_pd = pd.DataFrame(fund_grant_total_np2, columns=['日期', '沪深300', '累计收益率', '上证指数'])
# fund_grant_total_pd = pd.DataFrame(fund_grant_total_np2[:, [0, 2]], columns=['日期', 'b累计收益率'])
# # fund_grant_total_pd.to_csv('./data/temp/t2.csv', encoding='utf_8_sig', index=False)
#
# fund_grant_total_history_pd = pd.read_csv('./data/temp/t2.csv', low_memory=False)
# fund_grant_total_history_pd = pd.merge(fund_grant_total_history_pd,fund_grant_total_pd, how='left', on='日期')
# fund_grant_total_history_pd.to_csv('./data/temp/t3.csv', encoding='utf_8_sig', index=False)
#
#
import datetime
from dateutil.relativedelta import relativedelta
import time
import calendar
import arrow

#
# def get_quarter_end():
#     now = arrow.utcnow().to("local")
#     return now.ceil("quarter")

# print(get_quarter_end())


def getQuarterDateByStr():
    quarter_list = []
    for i in range(-1, -9, -1):
        # print('i', i)
        a = arrow.utcnow().to("local").shift(months=i*3)
        # print(a)
        # print(a.ceil("quarter").format('YYYY-MM-DD'))
        quarter_list.append(a.ceil("quarter").format('YYYY-MM-DD'))
    return quarter_list

getQuarterDateByStr()

# def add_months(dt,months):
#     #返回dt隔months个月后的日期，months相当于步长
#     month = dt.month - 1 + months
#     year = int(dt.year + month / 12)
#     month = month % 12 + 1
#     print(year, month)
#     day = min(dt.day, calendar.monthrange(year, month)[1])
#     return dt.replace(year=year, month=month, day=day)
#
# def getBetweenMonth(begin_date):
#     date_list = []
#     begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
#     end_date = datetime.datetime.strptime(time.strftime('%Y-%m-%d', time.localtime(time.time())), "%Y-%m-%d")
#     while begin_date <= end_date:
#         date_str = begin_date.strftime("%Y-%m")
#         date_list.append(date_str)
#         begin_date = add_months(begin_date, 1)
#     return date_list
#
# def getBetweenQuarter(begin_date):
#     quarter_list = []
#     month_list = getBetweenMonth(begin_date)
#     for value in month_list:
#         tempvalue = value.split("-")
#         if tempvalue[1] in ['01','02','03']:
#             quarter_list.append(tempvalue[0] + "Q1")
#         elif tempvalue[1] in ['04','05','06']:
#             quarter_list.append(tempvalue[0] + "Q2")
#         elif tempvalue[1] in ['07', '08', '09']:
#             quarter_list.append(tempvalue[0] + "Q3")
#         elif tempvalue[1] in ['10', '11', '12']:
#             quarter_list.append(tempvalue[0] + "Q4")
#     quarter_set = set(quarter_list)
#     quarter_list = list(quarter_set)
#     quarter_list.sort()
#     return quarter_list
#
# print(getBetweenQuarter('2017-01-01'))

# str = '基金投资风格2019年2季度'.replace('基金投资风格', '')
# year = str.split('年')[0] + '-'
# quarter = str.split('年')[1].split('季度')[0]
# quarterDate = ''
# if quarter == "1":
#     quarterDate = '3-31'
# elif quarter == "2":
#     quarterDate = '6-30'
# elif quarter == "3":
#     quarterDate = '9-30'
# elif quarter == "4":
#     quarterDate = '12-31'
# styleStr = year + quarterDate
# styleDate = datetime.datetime.strptime(styleStr, '%Y-%m-%d')
# print(styleStr)
# print(styleDate)
# print(styleDate.strftime("%Y-%m-%d"))

