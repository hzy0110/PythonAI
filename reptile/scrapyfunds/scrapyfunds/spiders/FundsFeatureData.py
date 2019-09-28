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
    # 抓取基金特色数据，季度级
    name = 'FundsManageFeatureData'  # 唯一，用于区别Spider。运行爬虫时，就要使用该名字
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
        if os.path.exists('./data/fund/feature_data_pd.csv'):
            os.remove('./data/fund/feature_data_pd.csv')

        url = 'http://fundf10.eastmoney.com/tsdata_{}.html'
        requests_list = []
        # Test
        # url_test = 'http://fundf10.eastmoney.com/tsdata_000006.html'
        # request = scrapy.Request(url_test, callback=self.parse_funds_list, meta={'code': '001968'})
        # requests_list.append(request)

        allCode = np.loadtxt("./data/fund/code_np.csv", dtype=np.str, delimiter=',').tolist()
        for code in allCode:
            request = scrapy.Request(url.format(code), callback=self.parse_funds_list, meta={'code': code})
            requests_list.append(request)

        return requests_list

    def parse_funds_list(self, response):
        code = response.meta['code']
        # 成立日期
        create_date = response.xpath('//div[@class="bs_gl"]/p/label/span/text()').get()
        # 公司代码
        company_code = response.xpath('//div[@class="bs_gl"]/p/label').css("a::attr(href)").getall()[-1].replace(
            "http://fund.eastmoney.com/company/", "").replace(".html", "")
        company_name = response.xpath('//div[@class="bs_gl"]/p/label/a/text()').getall()[-1]

        create_date_pd = pd.DataFrame([[code, company_code, company_name, create_date]],
                                      columns=['code', '公司code', '公司名称', '基金成立日期'])
        data_1 = response.xpath('//table[@class="fxtb"]')
        data_2 = response.xpath('//table[@class="fgtb"]')

        data_tr = data_1.xpath('tr')
        y1 = data_tr[0].xpath('th/text()').getall()[1]
        y2 = data_tr[0].xpath('th/text()').getall()[2]
        y3 = data_tr[0].xpath('th/text()').getall()[3]
        norm1 = data_tr[1].xpath('td/text()').getall()[0]
        norm2 = data_tr[2].xpath('td/text()').getall()[0]

        v1 = data_tr[1].xpath('td/text()').getall()[1]
        v2 = data_tr[1].xpath('td/text()').getall()[2]
        v3 = data_tr[1].xpath('td/text()').getall()[3]
        v4 = data_tr[2].xpath('td/text()').getall()[1]
        v5 = data_tr[2].xpath('td/text()').getall()[2]
        v6 = data_tr[2].xpath('td/text()').getall()[3]

        columns = [norm1 + y1, norm1 + y2, norm1 + y3, norm2 + y1, norm2 + y2, norm2 + y3]
        fund_feature_data_pd = pd.DataFrame([[v1, v2, v3, v4, v5, v6]], columns=columns)

        data_tr = data_2.xpath('tr')
        if len(data_tr) == 6:
            name = data_tr[0].xpath('th/text()').getall()[1]
            columns_list = []
            values_list = []
            for tr in data_tr:
                # qname = tr.xpath('td/text()').getall()[0]
                if len(tr.xpath('td/text()').getall()) > 0:
                    quarter_name = tr.xpath('td/text()').getall()[0]
                    values = tr.xpath('td/text()').getall()[1]
                    columns_list.append(name + quarter_name)
                    values_list.append(values)
            invest_style_pd = pd.DataFrame([values_list], columns=columns_list)
        elif len(data_tr) > 0:
            name = data_tr[0].xpath('th/text()').getall()[1]
            columns_list = []
            values_np = np.array(['', '', '', '', '', '', '', ''], dtype = np.object)
            # for tr in data_tr:
            for i in range(len(data_tr)):

                # qname = tr.xpath('td/text()').getall()[0]
                if len(data_tr[i].xpath('td/text()').getall()) > 0:
                    quarter_name = data_tr[i].xpath('td/text()').getall()[0]
                    values = data_tr[i].xpath('td/text()').getall()[1]
                    columns_list.append(name + quarter_name)
                    values_np[i-1] = values
            invest_style_pd = pd.DataFrame([values_np], columns=['基金投资风格2019年2季度', '基金投资风格2019年1季度', '基金投资风格2018年4季度',
                                                                 '基金投资风格2018年3季度', '基金投资风格2018年2季度', '基金投资风格2018年1季度',
                                                                 '基金投资风格2017年4季度', '基金投资风格2017年3季度'])
        else:
            invest_style_pd = pd.DataFrame([], columns=['基金投资风格2019年2季度', '基金投资风格2019年1季度', '基金投资风格2018年4季度',
                                                        '基金投资风格2018年3季度', '基金投资风格2018年2季度', '基金投资风格2018年1季度',
                                                        '基金投资风格2017年4季度', '基金投资风格2017年3季度'])

        # 合并 DataFrame
        feature_data_pd = pd.concat([create_date_pd, fund_feature_data_pd], axis=1)
        feature_data_pd = pd.concat([feature_data_pd, invest_style_pd], axis=1)
        # print("feature_data_pd", feature_data_pd, feature_data_pd.info())

        fundsItems = []
        fundsItem = ScrapyfundsItem()
        fundsItem['feature_data_pd'] = feature_data_pd
        fundsItems.append(fundsItem)
        return fundsItems
