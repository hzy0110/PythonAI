import scrapy
import pandas as pd
import os
from fake_useragent import UserAgent
import json
from scrapy.http import Request
from scrapyfunds.items import ScrapyfundsItem
import random
import numpy as np


class FundsSpider(scrapy.Spider):
    # 抓取同类信息，天级
    name = 'FundsGrandTotalReportPer'  # 唯一，用于区别Spider。运行爬虫时，就要使用该名字
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
        requests_list = []
        # dt = sixmont = 6 个月，all = 最大
        url = 'http://fund.eastmoney.com/data/FundPicData.aspx?bzdm={0}&n=0&dt=all&vname=ljsylSVG_PicData&r={1}'

        # if os.path.exists('./data/fund/fund_grant_total_all_pd.csv'):
        #     os.remove('./data/fund/fund_grant_total_all_pd.csv')
        request = scrapy.Request(url.format('040001', random.random()), callback=self.parse_funds_list,
                                 meta={'code': '040001'})
        requests_list.append(request)

        # else:
        #     request = scrapy.Request(url.format('040001', random.random()), callback=self.parse_funds_list,
        #                              meta={'code': '040001'})
        #     requests_list.append(request)

        allCode = np.loadtxt("./data/fund/code_np.csv", dtype=np.str, delimiter=',').tolist()
        for code in allCode:
            if code != '040001':
                request = scrapy.Request(url.format(code, random.random()), callback=self.parse_funds_list, meta={'code': code})
                requests_list.append(request)
        return requests_list

    def parse_funds_list(self, response):
        code = response.meta['code']
        fund_grant_total_str = response.body.decode('UTF-8')
        if len(fund_grant_total_str) > 24:
            fund_grant_total_str = fund_grant_total_str.replace('var ljsylSVG_PicData=', '')
            fund_grant_total_str = fund_grant_total_str.replace('"', '')

            fund_grant_total_list1 = fund_grant_total_str.split('|')
            # print(len(t1_list))
            fund_grant_total_list2 = []
            for s in fund_grant_total_list1:
                fund_grant_total_list2.append([s.split('_')])
            # print(len(t2_list))
            fund_grant_total_np1 = np.array(fund_grant_total_list2)
            # print(fund_grant_total_np1.shape)
            fund_grant_total_np2 = fund_grant_total_np1.reshape(fund_grant_total_np1.shape[0],
                                                                fund_grant_total_np1.shape[2])
            # print(fund_grant_total_np2.shape)

            # fund_grant_total_pd = pd.DataFrame(fund_grant_total_np2, columns=['日期', '沪深300', '累计收益率', '上证指数'])
            if code == '040001':
                # fund_grant_total_pd = pd.DataFrame(fund_grant_total_np2, columns=['日期', '沪深300', code + '|累计收益率', '上证指数'])
                fund_grant_total_pd = pd.DataFrame(fund_grant_total_np2, columns=['日期', '沪深300', code + '|累计收益率', '上证指数'])
            else:
                # if fund_grant_total_np2.shape[1] < 3:
                    # print("小于！！！！", code)
                fund_grant_total_pd = pd.DataFrame(fund_grant_total_np2[:, [0, 1]], columns=['日期', code + '|累计收益率'])
            # fund_grant_total_pd.to_csv('./data/temp/t2.csv', encoding='utf_8_sig', index=False)

            fundsItems = []
            fundsItem = ScrapyfundsItem()
            fundsItem['fund_grant_total_pd'] = fund_grant_total_pd
            fundsItem['code'] = code
            # fundsItem['manager_id'] = manager_id
            fundsItems.append(fundsItem)
            return fundsItems
