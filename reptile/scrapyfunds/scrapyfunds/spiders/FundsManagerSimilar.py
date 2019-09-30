import scrapy
import pandas as pd
import os
import execjs
import requests
from fake_useragent import UserAgent
import json
from scrapy.http import Request
from scrapyfunds.items import ScrapyfundsItem
import numpy as np
import arrow

class FundsSpider(scrapy.Spider):
    # 抓取基金经理同类平均和同类排名，更换经理抓，起码月级吧
    name = 'FundsManagerSimilar'  # 唯一，用于区别Spider。运行爬虫时，就要使用该名字
    allowed_domains = ['fund.eastmoney.com']  # 允许访问的域

    # 初始url。在爬取从start_urls自动开始后，服务器返回的响应会自动传递给parse(self, response)方法。
    # 说明：该url可直接获取到所有基金的相关数据
    # start_url = ['http://fundact.eastmoney.com/banner/pg.html#ln']

    # custome_setting可用于自定义每个spider的设置，而setting.py中的都是全局属性的，当你的scrapy工程里有多个spider的时候这个custom_setting就显得很有用了
    # custome_setting = {
    #
    # }

    # spider中初始的request是通过调用 start_requests() 来获取的。 start_requests() 读取 start_urls 中的URL， 并以 parse 为回调函数生成 Request 。
    # 重写start_requests也就不会从start_urls generate Requests了
    def start_requests(self):
        url = 'http://fundf10.eastmoney.com/jjjl_{}.html'
        if os.path.exists('./data/manager/manager_history_similar_pd.csv'):
            os.remove('./data/manager/manager_history_similar_pd.csv')
        if os.path.exists('./data/fund/fund_history_manager_pd.csv'):
            os.remove('./data/fund/fund_history_manager_pd.csv')

        if not os.path.exists('./data/temp/manager_id_np.csv'):
            file = open('./data/temp/manager_id_np.csv', 'w')
            file.write("-999,-999")
            file.close()
            # manager_id_pd = pd.DataFrame(columns=['经理ID'])
            # manager_id_pd.to_csv('./data/temp/manager_id_np.csv')
        else:
            file = open('./data/temp/manager_id_np.csv', 'w+')  # 打开文件
            file.truncate()  # 清空文件内容
            file.write("-999,-999")
            file.close()

        # allFunds_pd = pd.read_csv('/Users/hzy/code/PycharmProjects/reptile/wide/fund_wide_pd.csv', low_memory=False)
        # allFunds_code_s = allFunds_pd['code']
        requests_list = []
        # Test
        # url_test = 'http://fundf10.eastmoney.com/jjjl_000001.html'
        # url_test = 'http://fundf10.eastmoney.com/jjjl_000009.html'
        # url_test = 'http://fundf10.eastmoney.com/jjjl_000011.html'
        # url_test = 'http://fundf10.eastmoney.com/jjjl_004828.html'
        # url_test = 'http://fundf10.eastmoney.com/jjjl_000076.html'
        url_test = 'http://fundf10.eastmoney.com/jjjl_002839.html'
        request = scrapy.Request(url_test, callback=self.parse_funds_list, meta={'fund_code': '000001'})
        requests_list.append(request)

        # allCode = np.loadtxt("./data/fund/code_np.csv", dtype=np.str, delimiter=',').tolist()
        # for fund_code in allCode:
        #     request = scrapy.Request(url.format(fund_code), callback=self.parse_funds_list, meta={'fund_code': fund_code})
        #     requests_list.append(request)

        return requests_list

    def parse_funds_list(self, response):
        fund_code = response.meta['fund_code']
        data_1 = response.xpath('//table[@class="w782 comm  jloff"]')[0]
        # data_thead_r = data_1.xpath('thead/tr/th/text()').getall()
        data_thead_r = ['job_start_date', 'job_end_date', 'manager_code', 'job_work_day_num', 'job_report']
        fund_history_manager_pd = pd.DataFrame(columns=data_thead_r)
        # print('data_thead_r', data_thead_r)
        data_tbody = data_1.xpath('tbody/tr')
        fund_history_manager_td_list = []
        for body_tr in data_tbody:
            tds = body_tr.xpath('td')
            href = tds[2].css("a::attr(href)").getall()
            manager_num = len(href)
            for i in range(manager_num):
                v0 = tds[0].xpath('text()').get()
                v1 = tds[1].xpath('text()').get()
                if v1 == '至今':
                    v1 = arrow.now().date()
                v2 = href[i].replace("http://fund.eastmoney.com/manager/", "").replace(".html", "")
                v3 = self.getTimedelta(v0, v1)
                if tds[4].xpath('text()').get() != '' and tds[4].xpath('text()').get() is not None:
                    v4 = float(tds[4].xpath('text()').get().replace('%', ''))/100
                else:
                    v4 = 0

                fund_history_manager_td_list.append([v0, v1, v2, v3, v4])
            # for td in tds:
            #     if td.xpath('a/text()').get() is not None:
            #         href = td.css("a::attr(href)").getall()
            #         a = td.xpath('a/text()').getall()
            #         id_name_list = []
            #         for i in range(len(href)):
            #             m_id = href[i].replace("http://fund.eastmoney.com/manager/", "").replace(".html", "")
            #             m_name = a[i]
            #             id_name_list.append([m_id, m_name])
            #         fund_history_manager_td_list.append(id_name_list)
            #     elif td.xpath('text()').get() is not None:
            #         t = td.xpath('text()').get()
            #         fund_history_manager_td_list.append(t)
            #     else:
            #         fund_history_manager_td_list.append('')
        # print('fund_history_manager_td_list', fund_history_manager_td_list)
        fund_history_manager_tr_pd = pd.DataFrame(fund_history_manager_td_list, columns=data_thead_r)
        fund_history_manager_pd = fund_history_manager_pd.append(fund_history_manager_tr_pd, ignore_index=True)

        fund_history_manager_pd.insert(0, 'fund_code', fund_code)
        # print('fund_history_manager_pd', fund_history_manager_pd)

        # print('fund_history_manager_pd', fund_history_manager_pd.loc[0, '基金经理'])
        # print("type:", type(fund_history_manager_pd.loc[0, '基金经理']))
        managers_list = []
        managers_id_list = []
        data_2 = response.xpath('//div[@class="jl_office"]/table')
        if fund_history_manager_pd.shape[0]  > 0:
            m_info = fund_history_manager_pd.loc[0, 'manager_code']
            if len(fund_history_manager_pd) > 0:
                for data in data_2:

                    # data_thead_r = data.xpath('thead/tr/th/text()').getall()
                    data_thead_r = ['fund_code', 'fund_name', 'fund_type', 'job_start_date', 'job_end_date',
                                    'job_work_day_num', 'job_report', 'similar_report', 'similar_ranking']
                    manager_history_pd = pd.DataFrame(columns=data_thead_r)
                    data_tbody = data.xpath('tbody/tr')
                    for body_tr in data_tbody:
                        tds = body_tr.xpath('td')
                        manager_history_td_list = []
                        for td in tds:
                            if td.xpath('a/text()').get() is not None:
                                a = td.xpath('a/text()').get()
                                manager_history_td_list.append(a)
                                # print("a",a)
                            elif td.xpath('text()').get() is not None:
                                t = td.xpath('text()').get()
                                manager_history_td_list.append(t)
                                # print("t", t)
                            else:
                                manager_history_td_list.append('')

                        manager_history_tr_pd = pd.DataFrame([manager_history_td_list], columns=data_thead_r)
                        manager_history_pd = manager_history_pd.append(manager_history_tr_pd, ignore_index=True)

                    manager_history_pd.insert(0, 'manager_code', m_info)
                    manager_history_pd['job_report'].fillna('0%', inplace=True)
                    manager_history_pd['job_report'].replace(to_replace=r'^\s*$', value='0%', regex=True, inplace=True)
                    manager_history_pd['similar_report'].fillna('0%', inplace=True)
                    manager_history_pd['similar_report'].replace(to_replace=r'^\s*$', value='0%', regex=True, inplace=True)
                    manager_history_pd['similar_ranking'].replace(to_replace='-|-', value='0|0', inplace=True)
                    print('manager_history_pd',manager_history_pd)
                    # print(manager_history_pd['similar_ranking'])
                    manager_history_pd['fund_type'] = manager_history_pd.apply(
                        lambda x: self.fund_type_2_num(x['fund_type']), axis=1)

                    manager_history_pd['job_end_date'] = manager_history_pd.apply(
                        lambda x: x['job_end_date'] if x['job_end_date'] != '至今' else arrow.now().date(), axis=1)

                    manager_history_pd['job_work_day_num'] = manager_history_pd.apply(
                        lambda x: self.getTimedelta(x['job_start_date'], x['job_end_date']), axis=1)

                    manager_history_pd['job_report'] = manager_history_pd.apply(
                        lambda x: float(x['job_report'].replace('%', ''))/100, axis=1)

                    manager_history_pd['similar_report'] = manager_history_pd.apply(
                        lambda x: float(x['similar_report'].replace('%', ''))/100, axis=1)

                    manager_history_pd['similar_count'] = manager_history_pd.apply(
                        lambda x: x['similar_ranking'].split('|')[1], axis=1)

                    manager_history_pd['similar_ranking'] = manager_history_pd.apply(
                        lambda x: x['similar_ranking'].split('|')[0], axis=1)

                    manager_history_pd.drop('fund_name', axis=1, inplace=True)
                    managers_list.append(manager_history_pd)
                    managers_id_list.append(m_info)

        fundsItems = []
        fundsItem = ScrapyfundsItem()
        fundsItem['fund_history_manager_pd'] = fund_history_manager_pd
        fundsItem['manager_pd'] = managers_list
        fundsItem['manager_id'] = managers_id_list
        # # fundsItem['manager_id'] = manager_id
        fundsItems.append(fundsItem)
        return fundsItems

    def getTimedelta(self, startDayStr, endDayStr):
        if endDayStr == '至今':
            endDay = arrow.now()
        else:
            endDay = arrow.get(endDayStr)
        startDay = arrow.get(startDayStr)
        timedelta = endDay - startDay
        return timedelta.days

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
