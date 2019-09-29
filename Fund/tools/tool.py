import pandas as pd

import time

import datetime
import tushare as ts
from dateutil.relativedelta import relativedelta


class Tool:
    def __init__(self):
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

    # 获取交易日期
    def getTradeDay(self, targetDay, alldays):
        tradingdays = alldays[alldays['isOpen'] == 1]  # 开盘日
        if targetDay in tradingdays['calendarDate'].values:
            return True
        else:
            return False

    # 获取交易日期列表
    def getTradeDayList(self, worth_pd):
        # 获取起始时间戳，用于交易日期
        strartDate_pd = worth_pd.loc[(worth_pd[1] == 202001) & (worth_pd[0] == '时间戳')]
        startDate_dt = datetime.datetime.fromtimestamp(int(strartDate_pd[2]) / 1000)

        alldays = ts.trade_cal()
        diffDay_list = []
        tradeDay_list = []

        diffDay_list.append((startDate_dt - relativedelta(months=+1)).strftime("%Y-%m-%d"))
        diffDay_list.append((startDate_dt - relativedelta(months=+3)).strftime("%Y-%m-%d"))
        diffDay_list.append((startDate_dt - relativedelta(months=+6)).strftime("%Y-%m-%d"))

        for y in range(1, 6, 1):
            diffDay_list.append((startDate_dt - relativedelta(years=+y)).strftime("%Y-%m-%d"))
        for targetDayStr in diffDay_list:

            #     print(targetDay)
            isTradeDay = False
            #     targetDayStr = ''
            while not isTradeDay:
                #     print(diffDay)
                #         targetDayStr = (datetime.datetime.now() - datetime.timedelta(days=diffDay)).strftime('%Y-%m-%d')
                #     print(targetDayStr)
                isTradeDay = self.getTradeDay(targetDayStr, alldays)
                if not isTradeDay:
                    targetDay = datetime.datetime.strptime(targetDayStr, '%Y-%m-%d')
                    targetDayStr = (targetDay - relativedelta(days=-1)).strftime("%Y-%m-%d")
            tradeDay_list.append(targetDayStr)

        # 获取时间戳和时间戳列表
        def timestr_to_13timestamp(dt):
            timearray = time.strptime(dt, '%Y-%m-%d')
            timestamp13 = int(time.mktime(timearray))
            return int(round(timestamp13 * 1000))

        tradeDay_timestamp_list = []
        for t in tradeDay_list:
            tradeDay_timestamp_list.append(timestr_to_13timestamp(t))
        #     print(tradeDay_list)
        return tradeDay_timestamp_list

    def fund_type_2_num(self, type_name):
        if type_name == 'ETF-场内':
            return 1
        elif type_name == '债券型':
            return 2
        elif type_name == '债券指数':
            return 3
        elif type_name == '其他创新':
            return 4
        elif type_name == '分级杠杆':
            return 5
        elif type_name == '固定收益':
            return 6
        elif type_name == '定开债券':
            return 7
        elif type_name == '混合-FOF':
            return 8
        elif type_name == '混合型':
            return 9
        elif type_name == '理财型':
            return 10
        elif type_name == '联接基金':
            return 11
        elif type_name == '股票型':
            return 12
        elif type_name == '股票指数':
            return 13
        elif type_name == '货币型':
            return 14

    def fund_invest_style(self, style_name):
        if style_name == '大盘价值':
            return 1
        elif style_name == '大盘平衡':
            return 2
        elif style_name == '大盘成长':
            return 3
        elif style_name == '中盘价值':
            return 4
        elif style_name == '中盘平衡':
            return 5
        elif style_name == '中盘成长':
            return 6
        elif style_name == '小盘价值':
            return 7
        elif style_name == '小盘平衡':
            return 8
        elif style_name == '小盘成长':
            return 9

    def replace_none(self, none_str):
        if none_str == '暂无数据' or none_str == '--' or none_str == 'None':
            return 0


    # def get_pingzhongdata_2_df(self, basic_info_pd, funds):
    #     # 数据合并到DF
    #     basic_info_pd.reset_index(drop=True)
    #
    #     # stockCodesDF = pd.DataFrame([[fund[0]]], columns=['基金持仓股票代码'])
    #     # zqCodesDF = pd.DataFrame([[fund[1]]], columns=['基金持仓债券代码'])
    #
    #     netWorthDF = pd.DataFrame(
    #         funds[0], columns=['trade_timestamp', 'net_worth', 'equity_return', 'unit_money'])
    #     ACWorthDF = pd.DataFrame(funds[1], columns=['trade_timestamp', 'ac_worth'])
    #     grandTotalDF = pd.DataFrame(funds[2], columns=['累计收益率名称', '累计收益率信息'])
    #     for i in range(len(grandTotalDF['累计收益率信息'])):
    #         grandTotalDF['累计收益率信息'][i] = pd.DataFrame(
    #             grandTotalDF['累计收益率信息'][i],
    #             columns=['trade_timestamp', grandTotalDF['累计收益率名称'][i] + '累计收益率'])
    #     rateInSimilarTypeDF = pd.DataFrame(funds[3], columns=['trade_timestamp', 'similar_ranking', 'sc'])
    #     rateInSimilarPersentDF = pd.DataFrame(funds[4], columns=['similar_ranking_per'])
    #     # fluctuationScaleDF = pd.DataFrame(
    #     #     fund[7], columns=['规模变动日期', "规模变动净资产亿元", "规模变动较上期环比"])
    #
    #     # 已平
    #     # holderStructureDF = pd.DataFrame(fund[8], columns=['持有信息日期', '持有信息'])
    #     # for i in range(len(holderStructureDF['持有信息'])):
    #     #     holderStructureDF['持有信息'][i] = pd.DataFrame(
    #     #         holderStructureDF['持有信息'][i], columns=["持有比名称", "比例"])
    #     # holderStructureDF = holderStructureDF.T
    #     # holderStructureDF.reset_index(drop=True, inplace=True)
    #     # holderStructureDF.columns = holderStructureDF[:1].values.tolist()
    #     # holderStructureDF = holderStructureDF.drop(0)
    #     # holderStructureDF = holderStructureDF.reset_index(drop=True)
    #
    #     # 已平
    #     # assetAllocationDF = pd.DataFrame(fund[9], columns=['名称', '资产配置名称'])
    #     # for i in range(len(assetAllocationDF['资产配置名称'])):
    #     #     assetAllocationDF['资产配置名称'][i] = pd.DataFrame(
    #     #         assetAllocationDF['资产配置名称'][i], columns=["时间", "比例"])
    #     # assetAllocationDF = assetAllocationDF.T
    #     # assetAllocationDF.reset_index(drop=True, inplace=True)
    #     # assetAllocationDF.columns = assetAllocationDF[:1].values.tolist()
    #     # assetAllocationDF = assetAllocationDF.drop(0)
    #     # assetAllocationDF = assetAllocationDF.reset_index(drop=True)
    #
    #     # 已平
    #     performanceEvaluationDF = pd.DataFrame(funds[5]).T
    #     performanceEvaluationDF = performanceEvaluationDF[0].apply(pd.Series)
    #     performanceEvaluationDF.columns = ["业绩评价名称", "分数", "描述"]
    #
    #     performanceEvaluationDF = performanceEvaluationDF.T
    #
    #     performanceEvaluationDF.reset_index(drop=True, inplace=True)
    #     performanceEvaluationDF.columns = performanceEvaluationDF[:
    #                                                               1].values.tolist(
    #     )
    #     performanceEvaluationDF = performanceEvaluationDF.drop(0)
    #     performanceEvaluationDF = performanceEvaluationDF.drop(2)
    #
    #     performanceEvaluationDF = performanceEvaluationDF.reset_index(drop=True)
    #     if performanceEvaluationDF.shape[1] < 6:
    #         #         收益率放到稳定性前面
    #         syl = performanceEvaluationDF['收益率']
    #         performanceEvaluationDF.drop(labels=['收益率'], axis=1, inplace=True)
    #         performanceEvaluationDF.insert(1, '收益率', syl)
    #         #     插入选证能力
    #         performanceEvaluationDF.insert(loc=1, column='选证能力', value=6)
    #         #     插入抗风险
    #         performanceEvaluationDF.insert(loc=3, column='抗风险', value=6)
    #
    #     performanceEvaluationDF.columns = ['avg_score', 'choose_score', 'earn_per_score', 'resist_risk_score',
    #                                        'stable_score', 'manage_scale_score']
    #     performanceEvaluationDF['avg_score'] = self.replace_none(performanceEvaluationDF['avg_score'][0])
    #     performanceEvaluationDF['choose_score'] = self.replace_none(performanceEvaluationDF['choose_score'][0])
    #     performanceEvaluationDF['earn_per_score'] = self.replace_none(performanceEvaluationDF['earn_per_score'][0])
    #     performanceEvaluationDF['resist_risk_score'] = self.replace_none(
    #         performanceEvaluationDF['resist_risk_score'][0])
    #     performanceEvaluationDF['stable_score'] = self.replace_none(performanceEvaluationDF['stable_score'][0])
    #     performanceEvaluationDF['manage_scale_score'] = self.replace_none(
    #         performanceEvaluationDF['manage_scale_score'][0])
    #
    #     # 现任基金经理
    #     power_pd = pd.DataFrame()
    #     profit_pd = pd.DataFrame()
    #     currentFundManagerDF = pd.DataFrame(
    #         funds[6],
    #         columns=[
    #             'id', '姓名', '星级', '工作时间', '资金/基金数', '平均分', '截止日期', '评分', '收益能力'
    #         ])
    #
    #     for i in range(len(currentFundManagerDF["id"])):
    #         power = pd.DataFrame(currentFundManagerDF['评分'][i]).T
    #         power = power.drop(1)  # 删除描述
    #         power.columns = power[:1].values.tolist()  # 重命名列名
    #         power = power.drop(0)
    #         power.reset_index(drop=True, inplace=True)
    #         power_pd = pd.concat([power_pd, power])
    #         power_pd.reset_index(drop=True, inplace=True)
    #
    #         profit = pd.DataFrame(currentFundManagerDF['收益能力'][i]).T
    #         profit.columns = profit[:1].values.tolist()  # 重命名列名
    #         profit = profit.drop(0)
    #         profit.reset_index(drop=True, inplace=True)
    #         profit_pd = pd.concat([profit_pd, profit])
    #         profit_pd.reset_index(drop=True, inplace=True)
    #
    #     currentFundManagerDF = pd.concat([currentFundManagerDF, power_pd], axis=1)
    #     currentFundManagerDF = pd.concat([currentFundManagerDF, profit_pd], axis=1)
    #     col_list = currentFundManagerDF.columns.tolist()
    #     col_new_list = []
    #     for c_name in col_list:
    #         # print('c_name', str(c_name), type(str(c_name)))
    #         c_name_new = str(c_name).replace('(', '').replace(')', '').replace(',', '').replace('\'', '')
    #         col_new_list.append(c_name_new)
    #     # print('col_new_list', col_new_list)
    #     currentFundManagerDF.columns = col_new_list
    #     currentFundManagerDF = currentFundManagerDF.drop(['姓名', '工作时间', '截止日期', '任期收益'], axis=1)
    #     currentFundManagerDF = currentFundManagerDF.drop(['评分', '收益能力'], axis=1)
    #     # print('currentFundManagerDF', currentFundManagerDF.T)
    #     # print('currentFundManagerDF', currentFundManagerDF.columns.tolist())
    #     # print('currentFundManagerDF', currentFundManagerDF.shape)
    #     if currentFundManagerDF.shape[1] == 13:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_influence_score', 'manager_resist_risk_score',
    #                                         'manager_choice_score', 'manager_report_score', 'manager_stable_score',
    #                                         'manager_manage_scale_score', 'manager_experience_score',
    #                                         'manager_similar_mean_score', 'manager_hs300_score']
    #     elif currentFundManagerDF.shape[1] == 14:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_resist_risk_score',
    #                                         'manager_choice_score', 'manager_report_score', 'manager_stable_score',
    #                                         'manager_manage_scale_score', 'manager_experience_score',
    #                                         'manager_over_earn_score ',
    #                                         'manager_tracking_error_score', 'manager_similar_mean_score',
    #                                         'manager_hs300_score']
    #     elif currentFundManagerDF.shape[1] == 12:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_resist_risk_score',
    #                                         'manager_choice_score', 'manager_report_score', 'manager_stable_score',
    #                                         'manager_manage_scale_score', 'manager_experience_score',
    #                                         'manager_similar_mean_score', 'manager_hs300_score']
    #     elif currentFundManagerDF.shape[1] == 10:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_experience_score',
    #                                         'manager_report_score', 'manager_resist_risk_score', 'manager_stable_score',
    #                                         'manager_choice_score', 'manager_similar_mean_score']
    #
    #     elif currentFundManagerDF.shape[1] == 9:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_experience_score',
    #                                         'manager_report_score', 'manager_resist_risk_score', 'manager_stable_score',
    #                                         'manager_choice_score']
    #
    #     else:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_experience_score',
    #                                         'manager_report_score', 'manager_resist_risk_score', 'manager_stable_score',
    #                                         'manager_choice_score', 'manager_similar_mean_score', 'manager_hs300_score']
    #     # print('currentFundManagerDF', currentFundManagerDF.T)
    #
    #     #     print(currentFundManagerDF)
    #     #     print(currentFundManagerDF.shape)
    #     #     print(currentFundManagerDF.info())
    #     #     currentFundManagerDF.to_csv(
    #     #             './currentFundManagerDF' + mid + '.csv',
    #     #             encoding='utf_8_sig',
    #     #             index=False)
    #
    #     # 申购赎回 # 已平
    #     # buySedemptionDF = pd.DataFrame(fund[12], columns=['申购赎回日期', '申购赎回信息'])
    #     # for i in range(len(buySedemptionDF['申购赎回信息'])):
    #     #     buySedemptionDF['申购赎回信息'][i] = pd.DataFrame(
    #     #         buySedemptionDF['申购赎回信息'][i], columns=["时间", "份额"])
    #     # buySedemptionDF = buySedemptionDF.T
    #     # buySedemptionDF.reset_index(drop=True, inplace=True)
    #     # buySedemptionDF.columns = buySedemptionDF[:1].values.tolist()
    #     # buySedemptionDF = buySedemptionDF.drop(0)
    #     # buySedemptionDF = buySedemptionDF.reset_index(drop=True)
    #
    #     # 合并到最终宽窄两表
    #     # 建立宽表 合并
    #     fund_wide_pd = pd.DataFrame()
    #     fund_wide_pd = pd.concat([fund_wide_pd, basic_info_pd], axis=1)
    #     # fund_wide_pd = pd.concat([fund_wide_pd, stockCodesDF], axis=1)  #
    #     # fund_wide_pd = pd.concat([fund_wide_pd, zqCodesDF], axis=1)  #
    #     # fund_wide_pd = pd.concat([
    #     #     fund_wide_pd,
    #     #     pd.DataFrame([[fluctuationScaleDF["规模变动日期"].values.tolist()]],
    #     #                  columns=["规模变动日期"])
    #     # ],axis=1)
    #
    #     # fund_wide_pd = pd.concat([
    #     #     fund_wide_pd,
    #     #     pd.DataFrame([[fluctuationScaleDF["规模变动净资产亿元"].values.tolist()]],
    #     #                  columns=["规模变动净资产亿元"])
    #     # ], axis=1)
    #
    #     # fund_wide_pd = pd.concat([
    #     #     fund_wide_pd,
    #     #     pd.DataFrame([[fluctuationScaleDF["规模变动较上期环比"].values.tolist()]],
    #     #                  columns=["规模变动较上期环比"])
    #     # ], axis=1)
    #     # fund_wide_pd = pd.concat([fund_wide_pd, holderStructureDF], axis=1)
    #     # fund_wide_pd = pd.concat([fund_wide_pd, assetAllocationDF], axis=1)
    #     fund_wide_pd = pd.concat([fund_wide_pd, performanceEvaluationDF], axis=1)
    #     # fund_wide_pd = pd.concat([fund_wide_pd, currentFundManagerDF], axis=1)
    #     # fund_wide_pd.rename(columns={0: "经理"}, inplace=True)
    #     # fund_wide_pd = pd.concat([fund_wide_pd, buySedemptionDF], axis=1)
    #
    #     # 窄表
    #     rateInSimilar_pd = pd.DataFrame()
    #     rateInSimilar_pd = pd.concat([rateInSimilar_pd, rateInSimilarTypeDF], axis=1)
    #     rateInSimilar_pd = pd.concat([rateInSimilar_pd, rateInSimilarPersentDF], axis=1)
    #     # rateInSimilar_pd = rateInSimilar_pd.T
    #     # rateInSimilar_pd.insert(0, 'code', code)
    #
    #     worth_pd = pd.DataFrame()
    #     worth_pd = pd.concat([worth_pd, netWorthDF], axis=1)
    #     worth_pd = pd.merge(worth_pd, ACWorthDF)
    #     # worth_pd = worth_pd.T
    #     # worth_pd.insert(0, 'code', code)
    #
    #     grandTotal_pd = pd.DataFrame()
    #     for i in range(len(grandTotalDF['累计收益率信息'])):
    #         if i == 0:
    #             grandTotal_pd = pd.concat([
    #                 grandTotal_pd, grandTotalDF['累计收益率信息'][i]], axis=1)
    #         else:
    #             grandTotal_pd = pd.merge(
    #                 grandTotal_pd, grandTotalDF['累计收益率信息'][i], how='outer')
    #     if not grandTotal_pd.empty:
    #         if grandTotal_pd.shape[1] == 3:
    #             grandTotal_pd.columns = ['trade_timestamp', 'hs300_grand_total', 'self_grand_total']
    #         else:
    #             grandTotal_pd.columns = ['trade_timestamp', 'hs300_grand_total', 'similar_mean_grand_total',
    #                                      'self_grand_total']
    #     # grandTotal_pd = grandTotal_pd.T
    #     # grandTotal_pd.insert(0, 'code', code)
    #
    #     return fund_wide_pd, rateInSimilar_pd, grandTotal_pd, worth_pd, currentFundManagerDF
    #
    # def get_hb_pingzhongdata_2_df(self, basic_info_pd, funds):
    #     # 数据合并到DF
    #     basic_info_pd.reset_index(drop=True)
    #
    #     # stockCodesDF = pd.DataFrame([[fund[0]]], columns=['基金持仓股票代码'])
    #     # zqCodesDF = pd.DataFrame([[fund[1]]], columns=['基金持仓债券代码'])
    #
    #     millionCopiesIncomeDF = pd.DataFrame(
    #         funds[0], columns=['trade_timestamp', 'million_copies_income'])
    #     sevenDaysYearIncomeDF = pd.DataFrame(funds[1], columns=['trade_timestamp', 'seven_daysyear_income'])
    #     grandTotalDF = pd.DataFrame(funds[2], columns=['累计收益率名称', '累计收益率信息'])
    #     for i in range(len(grandTotalDF['累计收益率信息'])):
    #         grandTotalDF['累计收益率信息'][i] = pd.DataFrame(
    #             grandTotalDF['累计收益率信息'][i],
    #             columns=['trade_timestamp', grandTotalDF['累计收益率名称'][i] + '累计收益率'])
    #     rateInSimilarTypeDF = pd.DataFrame(funds[3], columns=['trade_timestamp', 'similar_ranking', 'sc'])
    #     rateInSimilarPersentDF = pd.DataFrame(funds[4], columns=['similar_ranking_per'])
    #     # fluctuationScaleDF = pd.DataFrame(
    #     #     fund[7], columns=['规模变动日期', "规模变动净资产亿元", "规模变动较上期环比"])
    #
    #     # 已平
    #     # holderStructureDF = pd.DataFrame(fund[8], columns=['持有信息日期', '持有信息'])
    #     # for i in range(len(holderStructureDF['持有信息'])):
    #     #     holderStructureDF['持有信息'][i] = pd.DataFrame(
    #     #         holderStructureDF['持有信息'][i], columns=["持有比名称", "比例"])
    #     # holderStructureDF = holderStructureDF.T
    #     # holderStructureDF.reset_index(drop=True, inplace=True)
    #     # holderStructureDF.columns = holderStructureDF[:1].values.tolist()
    #     # holderStructureDF = holderStructureDF.drop(0)
    #     # holderStructureDF = holderStructureDF.reset_index(drop=True)
    #
    #     # 已平
    #     # assetAllocationDF = pd.DataFrame(fund[9], columns=['名称', '资产配置名称'])
    #     # for i in range(len(assetAllocationDF['资产配置名称'])):
    #     #     assetAllocationDF['资产配置名称'][i] = pd.DataFrame(
    #     #         assetAllocationDF['资产配置名称'][i], columns=["时间", "比例"])
    #     # assetAllocationDF = assetAllocationDF.T
    #     # assetAllocationDF.reset_index(drop=True, inplace=True)
    #     # assetAllocationDF.columns = assetAllocationDF[:1].values.tolist()
    #     # assetAllocationDF = assetAllocationDF.drop(0)
    #     # assetAllocationDF = assetAllocationDF.reset_index(drop=True)
    #
    #     # 已平
    #     performanceEvaluationDF = pd.DataFrame([[0, 0, 0, 0, 0, 0]],
    #                                            columns=['avg_score', 'choose_score', 'earn_per_score',
    #                                                     'resist_risk_score',
    #                                                     'stable_score', 'manage_scale_score'])
    #     # performanceEvaluationDF = pd.DataFrame(fund[5]).T
    #     # performanceEvaluationDF = performanceEvaluationDF[0].apply(pd.Series)
    #     # performanceEvaluationDF.columns = ["业绩评价名称", "分数", "描述"]
    #     #
    #     # performanceEvaluationDF = performanceEvaluationDF.T
    #     #
    #     # performanceEvaluationDF.reset_index(drop=True, inplace=True)
    #     # performanceEvaluationDF.columns = performanceEvaluationDF[:
    #     #                                                           1].values.tolist(
    #     # )
    #     # performanceEvaluationDF = performanceEvaluationDF.drop(0)
    #     # performanceEvaluationDF = performanceEvaluationDF.drop(2)
    #     #
    #     # performanceEvaluationDF = performanceEvaluationDF.reset_index(drop=True)
    #     # if performanceEvaluationDF.shape[1] < 6:
    #     #     #         收益率放到稳定性前面
    #     #     syl = performanceEvaluationDF['收益率']
    #     #     performanceEvaluationDF.drop(labels=['收益率'], axis=1, inplace=True)
    #     #     performanceEvaluationDF.insert(1, '收益率', syl)
    #     #     #     插入选证能力
    #     #     performanceEvaluationDF.insert(loc=1, column='选证能力', value=6)
    #     #     #     插入抗风险
    #     #     performanceEvaluationDF.insert(loc=3, column='抗风险', value=6)
    #     #
    #     # performanceEvaluationDF.columns = ['avg_score', 'choose_score', 'earn_per_score', 'resist_risk_score',
    #     #                                    'stable_score', 'manage_scale_score']
    #     # performanceEvaluationDF['avg_score'] = self.replace_none(performanceEvaluationDF['avg_score'][0])
    #     # performanceEvaluationDF['choose_score'] = self.replace_none(performanceEvaluationDF['choose_score'][0])
    #     # performanceEvaluationDF['earn_per_score'] = self.replace_none(performanceEvaluationDF['earn_per_score'][0])
    #     # performanceEvaluationDF['resist_risk_score'] = self.replace_none(performanceEvaluationDF['resist_risk_score'][0])
    #     # performanceEvaluationDF['stable_score'] = self.replace_none(performanceEvaluationDF['stable_score'][0])
    #     # performanceEvaluationDF['manage_scale_score'] = self.replace_none(performanceEvaluationDF['manage_scale_score'][0])
    #
    #     # 现任基金经理 已平,但有两行
    #     power_pd = pd.DataFrame()
    #     profit_pd = pd.DataFrame()
    #     currentFundManagerDF = pd.DataFrame(
    #         funds[6],
    #         columns=[
    #             'id', '姓名', '星级', '工作时间', '资金/基金数', '平均分', '截止日期', '评分', '收益能力'
    #         ])
    #
    #     for i in range(len(currentFundManagerDF["id"])):
    #         power = pd.DataFrame(currentFundManagerDF['评分'][i]).T
    #         power = power.drop(1)  # 删除描述
    #         power.columns = power[:1].values.tolist()  # 重命名列名
    #         power = power.drop(0)
    #         power.reset_index(drop=True, inplace=True)
    #         power_pd = pd.concat([power_pd, power])
    #         power_pd.reset_index(drop=True, inplace=True)
    #
    #         profit = pd.DataFrame(currentFundManagerDF['收益能力'][i]).T
    #         profit.columns = profit[:1].values.tolist()  # 重命名列名
    #         profit = profit.drop(0)
    #         profit.reset_index(drop=True, inplace=True)
    #         profit_pd = pd.concat([profit_pd, profit])
    #         profit_pd.reset_index(drop=True, inplace=True)
    #
    #     currentFundManagerDF = pd.concat([currentFundManagerDF, power_pd], axis=1)
    #     currentFundManagerDF = pd.concat([currentFundManagerDF, profit_pd], axis=1)
    #     col_list = currentFundManagerDF.columns.tolist()
    #     col_new_list = []
    #     for c_name in col_list:
    #         # print('c_name', str(c_name), type(str(c_name)))
    #         c_name_new = str(c_name).replace('(', '').replace(')', '').replace(',', '').replace('\'', '')
    #         col_new_list.append(c_name_new)
    #     # print('col_new_list', col_new_list)
    #     currentFundManagerDF.columns = col_new_list
    #     currentFundManagerDF = currentFundManagerDF.drop(['姓名', '工作时间', '截止日期', '任期收益'], axis=1)
    #     currentFundManagerDF = currentFundManagerDF.drop(['评分', '收益能力'], axis=1)
    #     # print('currentFundManagerDF', currentFundManagerDF.T)
    #     # print('currentFundManagerDF', currentFundManagerDF.columns.tolist())
    #     if currentFundManagerDF.shape[1] == 13:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_influence_score', 'manager_resist_risk_score',
    #                                         'manager_choice_score', 'manager_report_score', 'manager_stable_score',
    #                                         'manager_manage_scale_score', 'manager_experience_score',
    #                                         'manager_similar_mean_score', 'manager_hs300_score']
    #     elif currentFundManagerDF.shape[1] == 14:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_resist_risk_score',
    #                                         'manager_choice_score', 'manager_report_score', 'manager_stable_score',
    #                                         'manager_manage_scale_score', 'manager_experience_score',
    #                                         'manager_over_earn_score ',
    #                                         'manager_tracking_error_score', 'manager_similar_mean_score',
    #                                         'manager_hs300_score']
    #
    #     elif currentFundManagerDF.shape[1] == 12:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_resist_risk_score', 'manager_choice_score', 'manager_report_score',
    #                                         'manager_stable_score', 'manager_manage_scale_score',
    #                                         'manager_experience_score',
    #                                         'manager_similar_mean_score', 'manager_hs300_score']
    #     elif currentFundManagerDF.shape[1] == 10:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_experience_score',
    #                                         'manager_report_score', 'manager_influence_score', 'manager_stable_score',
    #                                         'manager_choice_score', 'manager_similar_mean_score']
    #     else:
    #         currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
    #                                         'manager_experience_score',
    #                                         'manager_report_score', 'manager_resist_risk_score', 'manager_stable_score',
    #                                         'manager_choice_score', 'manager_similar_mean_score', 'manager_hs300_score']
    #
    #     #     print(currentFundManagerDF)
    #     #     print(currentFundManagerDF.shape)
    #     #     print(currentFundManagerDF.info())
    #     #     currentFundManagerDF.to_csv(
    #     #             './currentFundManagerDF' + mid + '.csv',
    #     #             encoding='utf_8_sig',
    #     #             index=False)
    #
    #     # 申购赎回 # 已平
    #     # buySedemptionDF = pd.DataFrame(fund[12], columns=['申购赎回日期', '申购赎回信息'])
    #     # for i in range(len(buySedemptionDF['申购赎回信息'])):
    #     #     buySedemptionDF['申购赎回信息'][i] = pd.DataFrame(
    #     #         buySedemptionDF['申购赎回信息'][i], columns=["时间", "份额"])
    #     # buySedemptionDF = buySedemptionDF.T
    #     # buySedemptionDF.reset_index(drop=True, inplace=True)
    #     # buySedemptionDF.columns = buySedemptionDF[:1].values.tolist()
    #     # buySedemptionDF = buySedemptionDF.drop(0)
    #     # buySedemptionDF = buySedemptionDF.reset_index(drop=True)
    #
    #     # 合并到最终宽窄两表
    #     # 建立宽表 合并
    #     fund_wide_pd = pd.DataFrame()
    #     fund_wide_pd = pd.concat([fund_wide_pd, basic_info_pd], axis=1)
    #     # fund_wide_pd = pd.concat([fund_wide_pd, stockCodesDF], axis=1)  #
    #     # fund_wide_pd = pd.concat([fund_wide_pd, zqCodesDF], axis=1)  #
    #     # fund_wide_pd = pd.concat([
    #     #     fund_wide_pd,
    #     #     pd.DataFrame([[fluctuationScaleDF["规模变动日期"].values.tolist()]],
    #     #                  columns=["规模变动日期"])
    #     # ],axis=1)
    #
    #     # fund_wide_pd = pd.concat([
    #     #     fund_wide_pd,
    #     #     pd.DataFrame([[fluctuationScaleDF["规模变动净资产亿元"].values.tolist()]],
    #     #                  columns=["规模变动净资产亿元"])
    #     # ], axis=1)
    #
    #     # fund_wide_pd = pd.concat([
    #     #     fund_wide_pd,
    #     #     pd.DataFrame([[fluctuationScaleDF["规模变动较上期环比"].values.tolist()]],
    #     #                  columns=["规模变动较上期环比"])
    #     # ], axis=1)
    #     # fund_wide_pd = pd.concat([fund_wide_pd, holderStructureDF], axis=1)
    #     # fund_wide_pd = pd.concat([fund_wide_pd, assetAllocationDF], axis=1)
    #     fund_wide_pd = pd.concat([fund_wide_pd, performanceEvaluationDF], axis=1)
    #     # fund_wide_pd = pd.concat([fund_wide_pd, currentFundManagerDF], axis=1)
    #     # fund_wide_pd.rename(columns={0: "经理"}, inplace=True)
    #     # fund_wide_pd = pd.concat([fund_wide_pd, buySedemptionDF], axis=1)
    #
    #     # 窄表
    #     rateInSimilar_pd = pd.DataFrame()
    #     rateInSimilar_pd = pd.concat([rateInSimilar_pd, rateInSimilarTypeDF], axis=1)
    #     rateInSimilar_pd = pd.concat([rateInSimilar_pd, rateInSimilarPersentDF], axis=1)
    #     # rateInSimilar_pd = rateInSimilar_pd.T
    #     # rateInSimilar_pd.insert(0, 'code', code)
    #
    #     income_pd = pd.DataFrame()
    #     income_pd = pd.concat([income_pd, millionCopiesIncomeDF], axis=1)
    #     income_pd = pd.merge(income_pd, sevenDaysYearIncomeDF)
    #     # worth_pd = worth_pd.T
    #     # worth_pd.insert(0, 'code', code)
    #
    #     grandTotal_pd = pd.DataFrame()
    #     for i in range(len(grandTotalDF['累计收益率信息'])):
    #         if i == 0:
    #             grandTotal_pd = pd.concat([
    #                 grandTotal_pd, grandTotalDF['累计收益率信息'][i]], axis=1)
    #         else:
    #             grandTotal_pd = pd.merge(
    #                 grandTotal_pd, grandTotalDF['累计收益率信息'][i], how='outer')
    #     if not grandTotal_pd.empty:
    #         if grandTotal_pd.shape[1] == 3:
    #             grandTotal_pd.columns = ['trade_timestamp', 'hs300_grand_total', 'self_grand_total']
    #         else:
    #             grandTotal_pd.columns = ['trade_timestamp', 'hs300_grand_total', 'similar_mean_grand_total',
    #                                      'self_grand_total']
    #     # grandTotal_pd = grandTotal_pd.T
    #     # grandTotal_pd.insert(0, 'code', code)
    #
    #     return fund_wide_pd, rateInSimilar_pd, grandTotal_pd, income_pd, currentFundManagerDF

