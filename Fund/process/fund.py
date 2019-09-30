import pandas as pd


class ProcessFund:
    def get_pingzhongdata_2_df(self, basic_info_pd, funds):
        # 数据合并到DF
        basic_info_pd.reset_index(drop=True)

        netWorthDF = pd.DataFrame(
            funds[0], columns=['trade_timestamp', 'net_worth', 'equity_return', 'unit_money'])
        ACWorthDF = pd.DataFrame(funds[1], columns=['trade_timestamp', 'ac_worth'])
        grandTotalDF = pd.DataFrame(funds[2], columns=['累计收益率名称', '累计收益率信息'])
        for i in range(len(grandTotalDF['累计收益率信息'])):
            grandTotalDF['累计收益率信息'][i] = pd.DataFrame(
                grandTotalDF['累计收益率信息'][i],
                columns=['trade_timestamp', grandTotalDF['累计收益率名称'][i] + '累计收益率'])
        rateInSimilarTypeDF = pd.DataFrame(funds[3], columns=['trade_timestamp', 'similar_ranking', 'sc'])
        rateInSimilarPersentDF = pd.DataFrame(funds[4], columns=['similar_ranking_per'])

        # 已平
        performanceEvaluationDF = pd.DataFrame(funds[5]).T
        performanceEvaluationDF = performanceEvaluationDF[0].apply(pd.Series)
        performanceEvaluationDF.columns = ["业绩评价名称", "分数", "描述"]

        performanceEvaluationDF = performanceEvaluationDF.T

        performanceEvaluationDF.reset_index(drop=True, inplace=True)
        performanceEvaluationDF.columns = performanceEvaluationDF[:
                                                                  1].values.tolist(
        )
        performanceEvaluationDF = performanceEvaluationDF.drop(0)
        performanceEvaluationDF = performanceEvaluationDF.drop(2)

        performanceEvaluationDF = performanceEvaluationDF.reset_index(drop=True)
        if performanceEvaluationDF.shape[1] < 6:
            #         收益率放到稳定性前面
            syl = performanceEvaluationDF['收益率']
            performanceEvaluationDF.drop(labels=['收益率'], axis=1, inplace=True)
            performanceEvaluationDF.insert(1, '收益率', syl)
            #     插入选证能力
            performanceEvaluationDF.insert(loc=1, column='选证能力', value=6)
            #     插入抗风险
            performanceEvaluationDF.insert(loc=3, column='抗风险', value=6)

        performanceEvaluationDF.columns = ['avg_score', 'choose_score', 'earn_per_score', 'resist_risk_score',
                                           'stable_score', 'manage_scale_score']
        performanceEvaluationDF['avg_score'] = self.replace_none(performanceEvaluationDF['avg_score'][0])
        performanceEvaluationDF['choose_score'] = self.replace_none(performanceEvaluationDF['choose_score'][0])
        performanceEvaluationDF['earn_per_score'] = self.replace_none(performanceEvaluationDF['earn_per_score'][0])
        performanceEvaluationDF['resist_risk_score'] = self.replace_none(
            performanceEvaluationDF['resist_risk_score'][0])
        performanceEvaluationDF['stable_score'] = self.replace_none(performanceEvaluationDF['stable_score'][0])
        performanceEvaluationDF['manage_scale_score'] = self.replace_none(
            performanceEvaluationDF['manage_scale_score'][0])

        # 现任基金经理
        power_pd = pd.DataFrame()
        profit_pd = pd.DataFrame()
        currentFundManagerDF = pd.DataFrame(
            funds[6],
            columns=[
                'id', '姓名', '星级', '工作时间', '资金/基金数', '平均分', '截止日期', '评分', '收益能力'
            ])

        for i in range(len(currentFundManagerDF["id"])):
            power = pd.DataFrame(currentFundManagerDF['评分'][i]).T
            power = power.drop(1)  # 删除描述
            power.columns = power[:1].values.tolist()  # 重命名列名
            power = power.drop(0)
            power.reset_index(drop=True, inplace=True)
            power_pd = pd.concat([power_pd, power])
            power_pd.reset_index(drop=True, inplace=True)

            profit = pd.DataFrame(currentFundManagerDF['收益能力'][i]).T
            profit.columns = profit[:1].values.tolist()  # 重命名列名
            profit = profit.drop(0)
            profit.reset_index(drop=True, inplace=True)
            profit_pd = pd.concat([profit_pd, profit])
            profit_pd.reset_index(drop=True, inplace=True)

        currentFundManagerDF = pd.concat([currentFundManagerDF, power_pd], axis=1)
        currentFundManagerDF = pd.concat([currentFundManagerDF, profit_pd], axis=1)
        col_list = currentFundManagerDF.columns.tolist()
        col_new_list = []
        for c_name in col_list:
            c_name_new = str(c_name).replace('(', '').replace(')', '').replace(',', '').replace('\'', '')
            col_new_list.append(c_name_new)
        currentFundManagerDF.columns = col_new_list
        currentFundManagerDF = currentFundManagerDF.drop(['姓名', '工作时间', '截止日期', '任期收益'], axis=1)
        currentFundManagerDF = currentFundManagerDF.drop(['评分', '收益能力'], axis=1)
        if currentFundManagerDF.shape[1] == 13:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_influence_score', 'manager_resist_risk_score',
                                            'manager_choice_score', 'manager_report_score', 'manager_stable_score',
                                            'manager_manage_scale_score', 'manager_experience_score',
                                            'manager_similar_mean_score', 'manager_hs300_score']
        elif currentFundManagerDF.shape[1] == 14:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_resist_risk_score',
                                            'manager_choice_score', 'manager_report_score', 'manager_stable_score',
                                            'manager_manage_scale_score', 'manager_experience_score',
                                            'manager_over_earn_score ',
                                            'manager_tracking_error_score', 'manager_similar_mean_score',
                                            'manager_hs300_score']
        elif currentFundManagerDF.shape[1] == 12:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_resist_risk_score',
                                            'manager_choice_score', 'manager_report_score', 'manager_stable_score',
                                            'manager_manage_scale_score', 'manager_experience_score',
                                            'manager_similar_mean_score', 'manager_hs300_score']
        elif currentFundManagerDF.shape[1] == 10:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_experience_score',
                                            'manager_report_score', 'manager_resist_risk_score', 'manager_stable_score',
                                            'manager_choice_score', 'manager_similar_mean_score']

        elif currentFundManagerDF.shape[1] == 9:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_experience_score',
                                            'manager_report_score', 'manager_resist_risk_score', 'manager_stable_score',
                                            'manager_choice_score']

        else:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_experience_score',
                                            'manager_report_score', 'manager_resist_risk_score', 'manager_stable_score',
                                            'manager_choice_score', 'manager_similar_mean_score', 'manager_hs300_score']

        # 合并到最终宽窄两表
        # 建立宽表 合并
        fund_wide_pd = pd.DataFrame()
        fund_wide_pd = pd.concat([fund_wide_pd, basic_info_pd], axis=1)
        fund_wide_pd = pd.concat([fund_wide_pd, performanceEvaluationDF], axis=1)

        # 窄表
        rateInSimilar_pd = pd.DataFrame()
        rateInSimilar_pd = pd.concat([rateInSimilar_pd, rateInSimilarTypeDF], axis=1)
        rateInSimilar_pd = pd.concat([rateInSimilar_pd, rateInSimilarPersentDF], axis=1)

        worth_pd = pd.DataFrame()
        worth_pd = pd.concat([worth_pd, netWorthDF], axis=1)
        worth_pd = pd.merge(worth_pd, ACWorthDF)

        grandTotal_pd = pd.DataFrame()
        for i in range(len(grandTotalDF['累计收益率信息'])):
            if i == 0:
                grandTotal_pd = pd.concat([
                    grandTotal_pd, grandTotalDF['累计收益率信息'][i]], axis=1)
            else:
                grandTotal_pd = pd.merge(
                    grandTotal_pd, grandTotalDF['累计收益率信息'][i], how='outer')
        if not grandTotal_pd.empty:
            if grandTotal_pd.shape[1] == 3:
                grandTotal_pd.columns = ['trade_timestamp', 'hs300_grand_total', 'self_grand_total']
            else:
                grandTotal_pd.columns = ['trade_timestamp', 'hs300_grand_total', 'similar_mean_grand_total',
                                         'self_grand_total']

        return fund_wide_pd, rateInSimilar_pd, grandTotal_pd, worth_pd, currentFundManagerDF

    def get_hb_pingzhongdata_2_df(self, basic_info_pd, funds):
        # 数据合并到DF
        basic_info_pd.reset_index(drop=True)
        millionCopiesIncomeDF = pd.DataFrame(
            funds[0], columns=['trade_timestamp', 'million_copies_income'])
        sevenDaysYearIncomeDF = pd.DataFrame(funds[1], columns=['trade_timestamp', 'seven_daysyear_income'])
        grandTotalDF = pd.DataFrame(funds[2], columns=['累计收益率名称', '累计收益率信息'])
        for i in range(len(grandTotalDF['累计收益率信息'])):
            grandTotalDF['累计收益率信息'][i] = pd.DataFrame(
                grandTotalDF['累计收益率信息'][i],
                columns=['trade_timestamp', grandTotalDF['累计收益率名称'][i] + '累计收益率'])
        rateInSimilarTypeDF = pd.DataFrame(funds[3], columns=['trade_timestamp', 'similar_ranking', 'sc'])
        rateInSimilarPersentDF = pd.DataFrame(funds[4], columns=['similar_ranking_per'])


        # 已平
        performanceEvaluationDF = pd.DataFrame([[0, 0, 0, 0, 0, 0]],
                                               columns=['avg_score', 'choose_score', 'earn_per_score',
                                                        'resist_risk_score',
                                                        'stable_score', 'manage_scale_score'])
        # 现任基金经理 已平,但有两行
        power_pd = pd.DataFrame()
        profit_pd = pd.DataFrame()
        currentFundManagerDF = pd.DataFrame(
            funds[6],
            columns=[
                'id', '姓名', '星级', '工作时间', '资金/基金数', '平均分', '截止日期', '评分', '收益能力'
            ])

        for i in range(len(currentFundManagerDF["id"])):
            power = pd.DataFrame(currentFundManagerDF['评分'][i]).T
            power = power.drop(1)  # 删除描述
            power.columns = power[:1].values.tolist()  # 重命名列名
            power = power.drop(0)
            power.reset_index(drop=True, inplace=True)
            power_pd = pd.concat([power_pd, power])
            power_pd.reset_index(drop=True, inplace=True)

            profit = pd.DataFrame(currentFundManagerDF['收益能力'][i]).T
            profit.columns = profit[:1].values.tolist()  # 重命名列名
            profit = profit.drop(0)
            profit.reset_index(drop=True, inplace=True)
            profit_pd = pd.concat([profit_pd, profit])
            profit_pd.reset_index(drop=True, inplace=True)

        currentFundManagerDF = pd.concat([currentFundManagerDF, power_pd], axis=1)
        currentFundManagerDF = pd.concat([currentFundManagerDF, profit_pd], axis=1)
        col_list = currentFundManagerDF.columns.tolist()
        col_new_list = []
        for c_name in col_list:
            c_name_new = str(c_name).replace('(', '').replace(')', '').replace(',', '').replace('\'', '')
            col_new_list.append(c_name_new)
        currentFundManagerDF.columns = col_new_list
        currentFundManagerDF = currentFundManagerDF.drop(['姓名', '工作时间', '截止日期', '任期收益'], axis=1)
        currentFundManagerDF = currentFundManagerDF.drop(['评分', '收益能力'], axis=1)
        if currentFundManagerDF.shape[1] == 13:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_influence_score', 'manager_resist_risk_score',
                                            'manager_choice_score', 'manager_report_score', 'manager_stable_score',
                                            'manager_manage_scale_score', 'manager_experience_score',
                                            'manager_similar_mean_score', 'manager_hs300_score']
        elif currentFundManagerDF.shape[1] == 14:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_resist_risk_score',
                                            'manager_choice_score', 'manager_report_score', 'manager_stable_score',
                                            'manager_manage_scale_score', 'manager_experience_score',
                                            'manager_over_earn_score ',
                                            'manager_tracking_error_score', 'manager_similar_mean_score',
                                            'manager_hs300_score']

        elif currentFundManagerDF.shape[1] == 12:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_resist_risk_score', 'manager_choice_score', 'manager_report_score',
                                            'manager_stable_score', 'manager_manage_scale_score',
                                            'manager_experience_score',
                                            'manager_similar_mean_score', 'manager_hs300_score']
        elif currentFundManagerDF.shape[1] == 10:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_experience_score',
                                            'manager_report_score', 'manager_influence_score', 'manager_stable_score',
                                            'manager_choice_score', 'manager_similar_mean_score']
        else:
            currentFundManagerDF.columns = ['manager_code', 'manager_star', 'manager_fund_num', 'manager_avg_score',
                                            'manager_experience_score',
                                            'manager_report_score', 'manager_resist_risk_score', 'manager_stable_score',
                                            'manager_choice_score', 'manager_similar_mean_score', 'manager_hs300_score']
        # 合并到最终宽窄两表
        # 建立宽表 合并
        fund_wide_pd = pd.DataFrame()
        fund_wide_pd = pd.concat([fund_wide_pd, basic_info_pd], axis=1)
        fund_wide_pd = pd.concat([fund_wide_pd, performanceEvaluationDF], axis=1)

        # 窄表
        rateInSimilar_pd = pd.DataFrame()
        rateInSimilar_pd = pd.concat([rateInSimilar_pd, rateInSimilarTypeDF], axis=1)
        rateInSimilar_pd = pd.concat([rateInSimilar_pd, rateInSimilarPersentDF], axis=1)

        income_pd = pd.DataFrame()
        income_pd = pd.concat([income_pd, millionCopiesIncomeDF], axis=1)
        income_pd = pd.merge(income_pd, sevenDaysYearIncomeDF)

        grandTotal_pd = pd.DataFrame()
        for i in range(len(grandTotalDF['累计收益率信息'])):
            if i == 0:
                grandTotal_pd = pd.concat([
                    grandTotal_pd, grandTotalDF['累计收益率信息'][i]], axis=1)
            else:
                grandTotal_pd = pd.merge(
                    grandTotal_pd, grandTotalDF['累计收益率信息'][i], how='outer')
        if not grandTotal_pd.empty:
            if grandTotal_pd.shape[1] == 3:
                grandTotal_pd.columns = ['trade_timestamp', 'hs300_grand_total', 'self_grand_total']
            else:
                grandTotal_pd.columns = ['trade_timestamp', 'hs300_grand_total', 'similar_mean_grand_total',
                                         'self_grand_total']

        return fund_wide_pd, rateInSimilar_pd, grandTotal_pd, income_pd, currentFundManagerDF
