import pandas as pd
import requests
import time
import execjs
from tools.tool import Tool




class Reptile:
    def __init__(self):
        pass


    # 获取 js 数据
    def get_pingzhongdata(self, fscode, type_name):
        type_id = Tool().fund_type_2_num(type_name)
        # 用requests获取到对应的文件
        content = requests.get(self.get_url(fscode))

        # 使用execjs获取到相应的数据
        jsContent = execjs.compile(content.text)
        if content.status_code != 200:
            return None, None, None, None
        #     print(type(jsContent))
        ishb = jsContent.eval('ishb')
        name = jsContent.eval('fS_name')
        code = jsContent.eval('fS_code')
        fund_sourceRate = jsContent.eval('fund_sourceRate')
        fund_Rate = jsContent.eval('fund_Rate')
        fund_minsg = jsContent.eval('fund_minsg')
        #     try:
        # stockCodes = jsContent.eval('stockCodes')
        #     except:
        #         stockCodes = ""
        # zqCodes = jsContent.eval('zqCodes')
        syl_1n = jsContent.eval('syl_1n')
        syl_6y = jsContent.eval('syl_6y')
        syl_3y = jsContent.eval('syl_3y')
        syl_1y = jsContent.eval('syl_1y')

        grandTotal = jsContent.eval('Data_grandTotal')  # 累计收益率走势
        rateInSimilarType = jsContent.eval('Data_rateInSimilarType')  # 同类排名走势
        rateInSimilarPersent = jsContent.eval(
            'Data_rateInSimilarPersent')  # 同类排名百分比
        # fluctuationScale = jsContent.eval(
        #     'Data_fluctuationScale')  # 规模变动 mom-较上期环比
        # holderStructure = jsContent.eval('Data_holderStructure')  # 持有人结构
        # assetAllocation = jsContent.eval('Data_assetAllocation')  # 资产配置

        currentFundManager = jsContent.eval(
            'Data_currentFundManager')  # 现任基金经理  多个表示有多个经理
        # buySedemption = jsContent.eval('Data_buySedemption')  # 申购赎回

        basic_info_pd = pd.DataFrame(data=
                                     [[name, type_id, ishb, fund_sourceRate, fund_Rate, fund_minsg,
                                       syl_1n, syl_6y, syl_3y, syl_1y]],
                                     columns=[
                                         "fund_name", "fund_type", 'ishb', "original_rate", "current_rate", "min_buy",
                                         "near_earn_1y_per", "near_earn_6m_per", "near_earn_3m_per", "near_earn_1m_per"
                                     ])

        # stockCodesList = []
        # zqCodesList = []

        grandTotalList = []
        rateInSimilarTypeList = []
        rateInSimilarPersentList = []
        # fluctuationScaleList = []
        # holderStructureList = []
        # assetAllocationList = []
        performanceEvaluationList = []
        currentFundManagerList = []
        # buySedemptionList = []

        # for sc in stockCodes[::-1]:
        #     stockCodesList.append(sc)
        # stockCodesList = stockCodes
        # for zq in zqCodes.split(","):
        #     zqCodesList.append(zq)

        if bool(ishb):
            millionCopiesIncome = jsContent.eval('Data_millionCopiesIncome')  # 每万份收益
            sevenDaysYearIncome = jsContent.eval('Data_sevenDaysYearIncome')  # 7日年化收益率
            millionCopiesIncomeList = []
            sevenDaysYearIncomeList = []

            for dayMllionCopiesIncome in millionCopiesIncome[::-1]:
                millionCopiesIncomeList.append([dayMllionCopiesIncome[0], dayMllionCopiesIncome[1]])

            for daySevenDaysYearIncome in sevenDaysYearIncome[::-1]:
                sevenDaysYearIncomeList.append([daySevenDaysYearIncome[0], daySevenDaysYearIncome[1]])
        else:
            netWorthTrend = jsContent.eval('Data_netWorthTrend')  # 单位净值走势
            aCWorthTrend = jsContent.eval('Data_ACWorthTrend')  # 累计净值走势
            netWorthList = []
            ACWorthList = []
            #  单位净值走势  提取各数组
            for dayWorth in netWorthTrend[::-1]:
                dayWorth['unitMoney'] = dayWorth['unitMoney'].replace('拆分：每份基金份额分拆', '')
                dayWorth['unitMoney'] = dayWorth['unitMoney'].replace('拆分：每份基金份额折算', '')
                dayWorth['unitMoney'] = dayWorth['unitMoney'].replace('分红：每份派现金', '')
                dayWorth['unitMoney'] = dayWorth['unitMoney'].replace('元', '')
                dayWorth['unitMoney'] = dayWorth['unitMoney'].replace('份', '')
                if dayWorth['unitMoney'] == '':
                    dayWorth['unitMoney'] = 0
                netWorthList.append([
                    dayWorth['x'], dayWorth['y'], dayWorth['equityReturn'], dayWorth['unitMoney']

                ])

            performanceEvaluation = jsContent.eval(
                'Data_performanceEvaluation'
            )  # 业绩评价 ['选股能力', '收益率', '抗风险', '稳定性','择时能力']
            #    业绩评价  ['选股能力', '收益率', '抗风险', '稳定性','择时能力']
            pe_avr = performanceEvaluation.get('avr')
            #     performanceEvaluationList.append(pe_avr)
            pe_c = performanceEvaluation.get('categories')
            pe_dsc = performanceEvaluation.get('dsc')
            pe_data = performanceEvaluation.get('data')
            if len(pe_c) == len(pe_data):
                pe_c_d_d = [["平均值", pe_avr, "平均值"]]
                for i in range(len(pe_c)):
                    pe_c_d_d.append([pe_c[i], pe_data[i], pe_dsc[i]])
            else:
                pe_c = ['选股能力', '收益率', '抗风险', '稳定性', '择时能力']
                pe_dsc = ['选股能力', '收益率', '抗风险', '稳定性', '择时能力']
                pe_data = [0, 0, 0, 0, 0]
                pe_c_d_d = [["平均值", pe_avr, "平均值"]]
                for i in range(len(pe_c)):
                    pe_c_d_d.append([pe_c[i], pe_data[i], pe_dsc[i]])
            performanceEvaluationList.append(pe_c_d_d)

            #   2*-1
            for dayACWorth in aCWorthTrend[::-1]:
                ACWorthList.append([dayACWorth[0], dayACWorth[1]])

        #   3*2*126
        for dayGrandTotal in grandTotal[::-1]:
            gname = dayGrandTotal['name']
            dataList = []
            for data in dayGrandTotal['data']:
                dataList.append([data[0], data[1]])
            grandTotalList.append([gname, dataList])

        for dayRateInSimilarType in rateInSimilarType[::-1]:
            rateInSimilarTypeList.append([
                dayRateInSimilarType['x'], dayRateInSimilarType['y'],
                dayRateInSimilarType['sc']
            ])

        for dayRateInSimilarPersent in rateInSimilarPersent[::-1]:
            rateInSimilarPersentList.append(dayRateInSimilarPersent[1])

        #     规模变动 环比，categories= 环比日期，y=净资产亿元， mom=净资产变动率
        # fs_c = fluctuationScale.get('categories')
        # fs_s = fluctuationScale.get('series')
        # for i in range(len(fs_c)):
        #     fluctuationScaleList.append([fs_c[i], fs_s[i]["y"], fs_s[i]["mom"]])

        #     持有人结构 每个持有比在四个日期下的具体比例，categories= 环比日期，name=名称，data=数据
        # hs_c = holderStructure.get('categories')
        # hs_s = holderStructure.get('series')
        # for i in range(len(hs_s)):
        #     hs_s_n = hs_s[i]["name"]
        #     hs_c_d = []
        #     for j in range(len(hs_c)):
        #         hs_c_d.append([hs_c[j], hs_s[i]['data'][j]])
        #     holderStructureList.append([hs_s_n, hs_c_d])

        #    资产配置 每个名称的配置再不同日期下的占净比，净资产额度
        # aa_s = assetAllocation.get('series')
        # aa_c = assetAllocation.get('categories')
        # for i in range(len(aa_s)):
        #     aa_s_n = aa_s[i]["name"]
        #     aa_c_d = []
        #     for j in range(len(aa_c)):
        #         aa_c_d.append([aa_c[j], aa_s[i]['data'][j]])
        #     assetAllocationList.append([aa_s_n, aa_c_d])

        #   现任基金经理
        for cfm in currentFundManager[::-1]:
            fm_id = cfm.get("id")
            fm_name = cfm.get("name")
            fm_star = cfm.get("star")
            fm_workTime = cfm.get("workTime")
            try:
                if cfm.get("fundSize") == '':
                    fm_fundSize = 0
                else:
                    fm_fundSize = cfm.get("fundSize").split('只')[0].split('(')[1] #  62.67亿(7只基金)
            except Exception as e:
                print("错误 code", code)
            #         power
            fm_power = cfm.get("power")
            fm_power_list = []
            fm_power_avr = fm_power.get('avr')
            if fm_power_avr == '暂无数据':
                fm_power_avr = 0
            fm_power_jzrq = fm_power.get('jzrq')  # 截止日期
            fm_c = fm_power.get('categories')
            if fm_c is None:
                fm_c = ["经验值", "收益率", "抗风险", "稳定性", "择时能力"]
            fm_dsc = fm_power.get('dsc')
            if fm_dsc is None:
                fm_dsc = ["经验值", "收益率", "抗风险", "稳定性", "择时能力"]
            fm_data = fm_power.get('data')
            if fm_data is None:
                fm_data = [0, 0, 0, 0, 0]
            for i in range(len(fm_c)):
                fm_power_list.append([fm_c[i], fm_dsc[i], fm_data[i]])

            #         profit
            fm_profit = cfm.get("profit")
            fm_profit_list = []
            fm_profit_c = fm_profit.get('categories')
            fm_profit_s_d = fm_profit.get('series')[0]["data"]
            for i in range(len(fm_profit_s_d)):
                fm_profit_list.append([fm_profit_c[i], fm_profit_s_d[i]["y"]])

            currentFundManagerList.append([
                fm_id, fm_name, fm_star, fm_workTime, fm_fundSize, fm_power_avr,
                fm_power_jzrq, fm_power_list, fm_profit_list
            ])

        #    申购赎回 每个日期的操作量
        # bs_s = buySedemption.get('series')
        # bs_c = buySedemption.get('categories')
        # for i in range(len(bs_s)):
        #     bs_s_n = bs_s[i]["name"]
        #     bs_c_d = []
        #     for j in range(len(bs_c)):
        #         bs_c_d.append([bs_c[j], bs_s[i]['data'][j]])
        #     buySedemptionList.append([bs_s_n, bs_c_d])

        # print(name, code)

        if bool(ishb):
            return code, bool(ishb), basic_info_pd, [
                millionCopiesIncomeList, sevenDaysYearIncomeList, grandTotalList,
                rateInSimilarTypeList, rateInSimilarPersentList, performanceEvaluationList,
                currentFundManagerList
            ]
        else:
            return code, bool(ishb), basic_info_pd, [
                netWorthList, ACWorthList, grandTotalList,
                rateInSimilarTypeList, rateInSimilarPersentList, performanceEvaluationList,
                currentFundManagerList
            ]

    def get_all_manage(self):
        url = 'http://fund.eastmoney.com/Data/FundDataPortfolio_Interface.aspx?dt=14&mc=returnjson&ft=all&pn=3000&pi=1&sc=abbname&st=asc'
        content = requests.get(url)
        jsContent = execjs.compile(content.text)
        rawData = jsContent.eval('returnjson')

        manager_current_pd = pd.DataFrame(columns=['manager_code', 'fund_code'])
        manager_info_pd = pd.DataFrame(columns=[
                'manager_code', 'manager_name', 'company_code', 'total_work_day',
                'current_fund_best_report_per', 'current_fund_best_report_code', 'current_fund_money', 'all_fund_best_report_per'
            ])
        # ["30634044","艾定飞","80053204","华商基金","007685,630005",
        #  "华商电子行业量化股票,华商动态阿尔法混合","295","14.97%","630005","华商动态阿尔法混合","10.50亿元","14.97%"]
        for manager in rawData['data']:
            allManage = []
            manager_code = manager[0]
            manager_name = manager[1]
            company_code = manager[2]
            # compName = manage[3]
            fund_code_list = manager[4].split(',')
            # fundName = manager[5]
            total_work_day = manager[6]
            current_fund_best_report_per = manager[7].replace('%', '').replace('--', '')
            current_fund_best_report_code = manager[8]
            current_fund_money = manager[10].replace('亿元', '').replace('--', '')
            all_fund_best_report_per = manager[11].replace('%', '').replace('--', '')
            allManage.append([
                manager_code, manager_name, company_code, total_work_day,
                current_fund_best_report_per, current_fund_best_report_code, current_fund_money, all_fund_best_report_per
            ])
            manager_info_pd = manager_info_pd.append(pd.DataFrame(allManage, columns=[
                'manager_code', 'manager_name', 'company_code', 'total_work_day',
                'current_fund_best_report_per', 'current_fund_best_report_code', 'current_fund_money', 'all_fund_best_report_per'
            ]), ignore_index=True)

            if len(fund_code_list) > 0:
                current_fund_pd = pd.DataFrame(fund_code_list, columns=['fund_code'])
                current_fund_pd.insert(0, 'manager_code', manager_code)
                manager_current_pd = manager_current_pd.append(current_fund_pd, ignore_index=True)
            # print(manager_info_pd.shape)
            # print(manager_current_pd.shape)
            # break
        return manager_info_pd, manager_current_pd

    def get_all_code(self):
        url = 'http://fund.eastmoney.com/js/fundcode_search.js'
        content = requests.get(url)
        jsContent = execjs.compile(content.text)
        rawData = jsContent.eval('r')
        allCode = []
        for code in rawData:
            allCode.append(code[0])
        return allCode

    def get_all_code_type(self):
        url = 'http://fund.eastmoney.com/js/fundcode_search.js'
        content = requests.get(url)
        jsContent = execjs.compile(content.text)
        rawData = jsContent.eval('r')
        allCode = []
        allCodeType = []
        for code in rawData:
            allCode.append(code[0])
            allCodeType.append(code[3])
        allCodeType_ditc = {
            'code': allCode,
            'type': allCodeType
        }
        return pd.DataFrame(allCodeType_ditc)

    def get_url(self, fscode):
        head = 'http://fund.eastmoney.com/pingzhongdata/'
        tail = '.js?v=' + time.strftime("%Y%m%d%H%M%S", time.localtime())
        return head + fscode + tail