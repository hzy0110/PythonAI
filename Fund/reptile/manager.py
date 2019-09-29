import pandas as pd
import requests
import execjs


class ReptileManager:
    def get_all_manager(self):
        url = 'http://fund.eastmoney.com/Data/FundDataPortfolio_Interface.aspx?dt=14&mc=returnjson&ft=all&pn=3000&pi=1&sc=abbname&st=asc'
        content = requests.get(url)
        jsContent = execjs.compile(content.text)
        rawData = jsContent.eval('returnjson')

        manager_current_pd = pd.DataFrame(columns=['manager_code', 'fund_code'])
        manager_info_pd = pd.DataFrame(columns=[
            'manager_code', 'manager_name', 'company_code', 'total_work_day',
            'current_fund_best_report_per', 'current_fund_best_report_code', 'current_fund_money',
            'all_fund_best_report_per'
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
                'current_fund_best_report_per', 'current_fund_best_report_code', 'current_fund_money',
                'all_fund_best_report_per'
            ]), ignore_index=True)

            if len(fund_code_list) > 0:
                current_fund_pd = pd.DataFrame(fund_code_list, columns=['fund_code'])
                current_fund_pd.insert(0, 'manager_code', manager_code)
                manager_current_pd = manager_current_pd.append(current_fund_pd, ignore_index=True)
            # print(manager_info_pd.shape)
            # print(manager_current_pd.shape)
            # break
        return manager_info_pd, manager_current_pd
