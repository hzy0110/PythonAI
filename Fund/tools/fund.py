import pandas as pd
from tools.tool import Tool
from tools.reptile import Reptile


class Fund:
    def __init__(self):
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
    # 以下方法未处理

    #  计算宽表所需通用方法
    def get_wide_parameter(self, code, colName, target_pd, calType, dateModel, tradeDay_timestamp_list):
        get_index_data = target_pd.loc[(target_pd[1] == code)]
        fund_data = get_index_data.loc[get_index_data[0] == colName]

        return_pd = pd.DataFrame()
        if fund_data.shape[0] > 0:
            # 排除前两列
            fund_data = fund_data.iloc[:, 2:]
            fund_data = fund_data.T
            get_index_data = get_index_data.T
            if calType != 'dividend':
                fund_data = fund_data[fund_data.iloc[:, 0] > 0]

            if calType == 'mean' and dateModel == 'year':
                year_mean_pd = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0]], columns=[
                    colName + '30天平均值',
                    colName + '90天平均值',
                    colName + '180天平均值',
                    colName + '1年平均值',
                    colName + '2年平均值',
                    colName + '3年平均值',
                    colName + '4年平均值',
                    colName + '5年平均值'
                ])
            elif calType == 'mean_self' and dateModel == 'month':
                year_mean_pd = pd.DataFrame([[0, 0, 0]], columns=[
                    'GT各自类型累计收益率' + '30天平均值',
                    'GT各自类型累计收益率' + '90天平均值',
                    'GT各自类型累计收益率' + '180天平均值'
                ])
            elif calType == 'mean' and dateModel == 'month':
                year_mean_pd = pd.DataFrame([[0, 0, 0]], columns=[
                    colName + '30天平均值',
                    colName + '90天平均值',
                    colName + '180天平均值'
                ])

            elif calType == 'dividend' and dateModel == 'year':
                colName = '分红'
                dividend_pd = pd.DataFrame()
                year_count_pd = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0]], columns=[
                    colName + '30天次数',
                    colName + '90天次数',
                    colName + '180天次数',
                    colName + '1年次数',
                    colName + '2年次数',
                    colName + '3年次数',
                    colName + '4年次数',
                    colName + '5年次数'
                ])

                year_amount_pd = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0]], columns=[
                    colName + '30天总金额',
                    colName + '90天总金额',
                    colName + '180天总金额',
                    colName + '1年总金额',
                    colName + '2年总金额',
                    colName + '3年总金额',
                    colName + '4年总金额',
                    colName + '5年总金额'
                ])
            elif calType == 'report' and dateModel == 'year':
                colName = '净值回报率'
                year_report_pd = pd.DataFrame(
                    [[0, 0, 0, 0, 0, 0, 0, 0]],
                    columns=[
                        '近30天' + colName,
                        '近90天' + colName,
                        '近180天' + colName,
                        '近1年' + colName,
                        '近2年' + colName,
                        '近3年' + colName,
                        '近4年' + colName,
                        '近5年' + colName
                    ])
            elif calType == 'diff' and dateModel == 'year':
                year_similar_diff_pd = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0]], columns=[
                    colName + '5年上升',
                    colName + '4年上升',
                    colName + '3年上升',
                    colName + '2年上升',
                    colName + '1年上升',
                    colName + '180天上升',
                    colName + '90天上升',
                    colName + '30天上升'
                ])

            #         i = 0;
            l = 8
            if dateModel == 'month':
                l = 3
            for i in range(l):
                get_index_data_1 = get_index_data[get_index_data.iloc[:, 0] == int(tradeDay_timestamp_list[i])]
                if not get_index_data_1.empty:
                    #                 要-1 才是正确位置
                    y = get_index_data_1.index[0] - 1
                    fund_data_y = fund_data.iloc[0:y, :]
                    if calType == 'mean':
                        mean = fund_data_y.mean(0)  # 求平均
                        year_mean_pd.iloc[:, i] = mean.values[0]
                    if calType == 'dividend':
                        count = fund_data_y.where(fund_data_y > 0).count()  # 求次数
                        amount = fund_data_y.sum()
                        #                     print('i:',i, 'count:',count.values[0], 'amount:',amount.values[0])
                        year_count_pd.iloc[:, i] = count
                        year_amount_pd.iloc[:, i] = amount.iloc[0]
                        dividend_pd = pd.concat([year_count_pd, year_amount_pd], axis=1)
                    if calType == 'report':
                        report = (fund_data_y.iloc[0, :] - fund_data_y.iloc[-1, :]) / fund_data_y.iloc[-1, :] * 100
                        year_report_pd.iloc[:, i] = report.iloc[0]
                    if calType == 'diff':
                        diff = fund_data_y.iloc[0, :] - fund_data_y.iloc[-1, :]
                        year_similar_diff_pd.iloc[:, i] = diff.iloc[0]

            if calType == 'mean':
                return_pd = year_mean_pd
            if calType == 'dividend':
                return_pd = dividend_pd
            if calType == 'report':
                return_pd = year_report_pd
            if calType == 'diff':
                return_pd = year_similar_diff_pd
            return return_pd
        else:
            return None

    # 综合添加
    def zhcl(self, worth_pd, rateInSimilar_pd, grandTotal_pd):

        startFileName = './wide/fund_wide_pd.csv'
        fund_wide_pd = pd.read_csv(startFileName, low_memory=False)
        # 交易日
        tradeDay_timestamp_list = Tool().getTradeDayList()

        allCode = Reptile().get_all_code()
        # allCode = ['0040001']
        # allCode = ['000003'] # 分红测试用

        pds_list = []
        colnames_list = []
        calType_list = []
        dateModel_list = []
        saveFileName_list = []

        # # 添加worth 和 rateInSimilar 表集合的均值
        # pds_list.append([worth_pd,rateInSimilar_pd,rateInSimilar_pd,rateInSimilar_pd])
        # colnames_list.append(['净值回报','同类排名数','sc','同类排名百分比'])
        # calType_list.append('mean')
        # dateModel_list.append('year')
        # saveFileName_list.append('./data/fund/fund_wide_avg_1_pd.csv')

        # # 添加 gt 表集合的均值
        # pds_list.append([grandTotal_pd,grandTotal_pd])
        # colnames_list.append(['沪深300累计收益率','同类平均累计收益率'])
        # calType_list.append('mean')
        # dateModel_list.append('month')
        # saveFileName_list.append('./data/fund/fund_wide_avg_2_pd.csv')

        # # 添加 grandTotal_pd最后行不定名称
        # pds_list.append([grandTotal_pd])
        # colnames_list.append([''])
        # calType_list.append('mean_self')
        # dateModel_list.append('month')
        # saveFileName_list.append('./data/fund/fund_wide_avg_gt_pd.csv')

        # 添加 分红数据
        startFileName = './data/fund/fund_wide_avg_gt_pd.csv'
        fund_wide_pd = pd.read_csv(startFileName, low_memory=False)

        pds_list.append([worth_pd])
        colnames_list.append(['每份派送金'])
        calType_list.append('dividend')
        dateModel_list.append('year')
        saveFileName_list.append('./data/fund/fund_wide_dividend_pd.csv')

        # # 添加回报率
        pds_list.append([worth_pd])
        colnames_list.append(['单位净值'])
        calType_list.append('report')
        dateModel_list.append('year')
        saveFileName_list.append('./data/fund/fund_wide_report_pd.csv')

        # # 添加排名升降
        pds_list.append([rateInSimilar_pd, rateInSimilar_pd])
        colnames_list.append(['同类排名数', 'sc'])
        calType_list.append('diff')
        dateModel_list.append('year')
        saveFileName_list.append('./data/fund/fund_wide_rating_diff_pd.csv')

        for x in range(len(pds_list)):
            for i in range(len(pds_list[x])):
                #         print('len(pds_list[x])',len(pds_list[x]),colnames_list[x][0])
                parameter_pd = pd.DataFrame()
                print('正在执行', calType_list[x], dateModel_list[x], '的', colnames_list[x][i])
                for code in allCode:
                    r_pd = None
                    if calType_list[x] == 'mean_self':
                        self_data = grandTotal_pd.loc[grandTotal_pd[1] == int(code)]
                        if not self_data.empty:
                            name = self_data.iloc[self_data.shape[0] - 1, 0]
                            colnames_list[x][i] = name
                            r_pd = self.get_wide_parameter(int(code), colnames_list[x][i], pds_list[x][i], calType_list[x],
                                                      dateModel_list[x])
                    else:
                        r_pd = self.get_wide_parameter(int(code), colnames_list[x][i], pds_list[x][i], calType_list[x],
                                                  dateModel_list[x])

                    if r_pd is not None:
                        r_pd.insert(0, 'code', int(code))
                        parameter_pd = parameter_pd.append(r_pd, ignore_index=True)
                #                 print('parameter_pd',parameter_pd)
                #                 parameter_pd.to_csv('./data/temp/parameter_pd.csv', encoding='utf_8_sig', index=False)
                # 合并均值和原始 wide 并保存
                print("fund_wide_pd", fund_wide_pd.shape)
                print("parameter_pd", parameter_pd.shape)
                #     print(mean_pd)
                fund_wide_pd = pd.merge(fund_wide_pd, parameter_pd, how='left', on='code')
                print("fund_wide_pd", fund_wide_pd.shape)

                print('完成第', calType_list[x], dateModel_list[x], '的', colnames_list[x][i])
            # fund_wide_pd.to_csv(saveFileName_list[x], encoding='utf_8_sig', index=False)
            # fund_wide_pd = pd.read_csv(saveFileName_list[x], low_memory=False)

    # 处理基金经理
    # 统计指标，成立以来有过 N 名经理，最近更换经理到现在的天数
    def get_recent_manage_info(self, code, fund_wide_pd):
        manage_single_pd = pd.DataFrame(columns=[
            '经理ID', ' 经理星级', '经理在这个基金工作时间', '经理资金数', '经理基金数', '经理平均分', '经理经验值',
            '经理收益率', '经理抗风险', '经理稳定性', '经理择时能力', '经理任期收益', '经理同类平均', '经理沪深300',
            '经理跟踪误差', '经理超额收益', '经理管理规模'
        ])
        manages = fund_wide_pd.loc[fund_wide_pd['code'] == int(code)]['经理']
        if manages.shape[0] > 0:
            manage_list = fund_wide_pd.loc[fund_wide_pd['code'] ==
                                           int(code)]['经理'].values[0].split('\n')
            manage_pd = pd.DataFrame()
            manage_c = len(manage_list[0].split())  # 经理人数

            manage_list[0] = '列名 ' + manage_list[0]
            # 把经理恢复成 pd，可操作
            for row in manage_list:
                row_s = row.split()
                series = pd.Series(row_s[1:], name=row_s[0]).T
                manage_pd = manage_pd.append(series)
            #         print(manage_pd)

            #         print(manage_pd.shape)
            #         print(manage_pd)
            if manage_pd.shape[0] == 3:
                print("空经理 code", code)
                return manage_single_pd
            manage_wordDay_Total = 0
            manage_id = '0'
            target = -1
            # 收益率
            #     找出时间最长的经理
            #     workDay_s = manage_pd.loc[manage_pd[0] == '工作时间']
            #     print("manage_c",manage_c)
            for i in range(manage_c):
                #             try:
                workDay_s = manage_pd.loc[['工作时间'], i]
                #             except:
                #                 return []
                manage_wordDay = workDay_s[0].split('又')
                #     起码1 年以上经历
                if len(manage_wordDay) > 1:
                    manage_word_year = manage_wordDay[0].split('年')[0]
                    manage_word_day = manage_wordDay[1].split('天')[0]
                    manage_wordDay_Total_new = int(manage_word_year) * 365 + int(
                        manage_word_day)
                    if manage_wordDay_Total_new > manage_wordDay_Total:
                        manage_wordDay_Total = manage_wordDay_Total_new
                        target = i

            #     根据最长经理所在列取数据

            if target != -1:
                target_s = manage_pd.loc[:, target]
                manage_id = target_s['id']
                manage_star = target_s['星级']
                if str(target_s['资金/基金数']) != 'nan':
                    manage_fund_money_total = target_s['资金/基金数'].split('亿')[
                        0]  # 管理总基金额
                    manage_funds_total = target_s['资金/基金数'].split('(')[1].split(
                        '只')[0]  # 管理总基金数量
                else:
                    manage_fund_money_total = 0
                    manage_funds_total = 0
                manage_point_avg = target_s['平均分']
                manage_exp = target_s['(经验值,)']
                manage_earn_per = target_s['(收益率,)']
                try:
                    manage_resist_risk = target_s['(抗风险,)']
                except:
                    manage_resist_risk = 0

                try:
                    manage_stable = target_s['(稳定性,)']
                except:
                    manage_stable = 0

                try:
                    manage_choice = target_s['(择时能力,)']
                except:
                    manage_choice = 0

                try:
                    manage_eran_term = target_s['(任期收益,)']
                except:
                    manage_eran_term = 0

                try:
                    manage_similar_avg = target_s['(同类平均,)']
                except:
                    manage_similar_avg = 0

                try:
                    manage_hs300 = target_s['(沪深300,)']
                except:
                    manage_hs300 = 0

                try:
                    manage_track_error = target_s['(跟踪误差,)']
                except:
                    manage_track_error = 0

                try:
                    manage_eran_excess = target_s['(超额收益,)']
                except:
                    manage_eran_excess = 0

                try:
                    manage_manage_scale = target_s['(管理规模,)']
                except:
                    manage_manage_scale = 0

                insertRow = pd.DataFrame(
                    [[
                        manage_id, manage_star, manage_wordDay_Total,
                        manage_fund_money_total, manage_funds_total,
                        manage_point_avg, manage_exp, manage_earn_per,
                        manage_resist_risk, manage_stable, manage_choice,
                        manage_eran_term, manage_similar_avg, manage_hs300,
                        manage_track_error, manage_eran_excess, manage_manage_scale
                    ]],
                    columns=[
                        '经理ID', ' 经理星级', '经理在这个基金工作时间', '经理资金数', '经理基金数', '经理平均分',
                        '经理经验值', '经理收益率', '经理抗风险', '经理稳定性', '经理择时能力', '经理任期收益',
                        '经理同类平均', '经理沪深300', '经理跟踪误差', '经理超额收益', '经理管理规模'
                    ])
                manage_single_pd = manage_single_pd.append(
                    insertRow, ignore_index=True)
                return manage_single_pd
            else:
                #             print("target=-1,code=",code)
                return manage_single_pd
        else:
            #         print("manages.shape[0]=0,code=",code)
            return manage_single_pd

    def dispose_manager(self, fund_wide_pd):
        fund_wide_manage_pd = pd.read_csv('./data/fund/fund_wide_rating_diff_pd.csv', low_memory=False)
        allCode = Reptile().get_all_code()
        # allCode = ['040001','003150']
        manages_pd = pd.DataFrame()
        for code in allCode:
            manage_s_pd = self.get_recent_manage_info(code)
            manage_s_pd.insert(0, 'code', int(code))
            manages_pd = manages_pd.append(manage_s_pd, ignore_index=True)

        # 合并经理和原始 wide 并保存
        print(fund_wide_pd.shape)
        fund_wide_manage_pd = pd.merge(fund_wide_manage_pd, manages_pd, how='left', on='code')
        print(fund_wide_pd.shape)
        print(manages_pd.shape)
        fund_wide_manage_pd.to_csv("./data/fund/fund_wide_manage_pd.csv", encoding='utf_8_sig', index=False)
        fund_wide_manage_pd = pd.read_csv('./data/fund/fund_wide_manage_pd.csv', low_memory=False)