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


class FundsSpider(scrapy.Spider):
    # 抓取基金经理同类平均和同类排名，更换经理抓，起码月级吧
    name = 'FundsManageSimilar'  # 唯一，用于区别Spider。运行爬虫时，就要使用该名字
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
        # request = scrapy.Request(url_test, callback=self.parse_funds_list, meta={'code': '000001'})
        # requests_list.append(request)

        allCode = np.loadtxt("./data/fund/code_np.csv", dtype=np.str, delimiter=',').tolist()
        for code in allCode:
            request = scrapy.Request(url.format(code), callback=self.parse_funds_list, meta={'code': code})
            requests_list.append(request)

        return requests_list

    def parse_funds_list(self, response):
        code = response.meta['code']
        data_1 = response.xpath('//table[@class="w782 comm  jloff"]')[0]
        data_thead_r = data_1.xpath('thead/tr/th/text()').getall()
        fund_history_manager_pd = pd.DataFrame(columns=data_thead_r)
        data_tbody = data_1.xpath('tbody/tr')
        for body_tr in data_tbody:
            tds = body_tr.xpath('td')
            fund_history_manager_td_list = []
            for td in tds:
                if td.xpath('a/text()').get() is not None:
                    href = td.css("a::attr(href)").getall()
                    a = td.xpath('a/text()').getall()
                    id_name_list = []
                    for i in range(len(href)):
                        m_id = href[i].replace("http://fund.eastmoney.com/manager/", "").replace(".html", "")
                        m_name = a[i]
                        id_name_list.append([m_id, m_name])
                    fund_history_manager_td_list.append(id_name_list)
                elif td.xpath('text()').get() is not None:
                    t = td.xpath('text()').get()
                    fund_history_manager_td_list.append(t)
                else:
                    fund_history_manager_td_list.append('')
            fund_history_manager_tr_pd = pd.DataFrame([fund_history_manager_td_list], columns=data_thead_r)
            fund_history_manager_pd = fund_history_manager_pd.append(fund_history_manager_tr_pd, ignore_index=True)
        fund_history_manager_pd.insert(0, 'code', code)
        # print(fund_history_manager_pd)

        # print('fund_history_manager_pd', fund_history_manager_pd.loc[0, '基金经理'])
        # print("type:", type(fund_history_manager_pd.loc[0, '基金经理']))
        managers_list = []
        managers_id_list = []
        data_2 = response.xpath('//div[@class="jl_office"]/table')
        if len(fund_history_manager_pd) > 0:
            for data, m_info in zip(data_2, fund_history_manager_pd.loc[0, '基金经理']):
                # print("m_info", m_info)
                data_thead_r = data.xpath('thead/tr/th/text()').getall()
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
                manager_history_pd.insert(0, '经理ID', m_info[0])
                # print("manager_history_pd", manager_history_pd)
                managers_list.append(manager_history_pd)
                managers_id_list.append(m_info[0])

        fundsItems = []
        fundsItem = ScrapyfundsItem()
        fundsItem['fund_history_manager_pd'] = fund_history_manager_pd
        fundsItem['manager_pd'] = managers_list
        fundsItem['manager_id'] = managers_id_list
        # # fundsItem['manager_id'] = manager_id
        fundsItems.append(fundsItem)
        return fundsItems
