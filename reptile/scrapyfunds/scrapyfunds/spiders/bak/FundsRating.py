import scrapy
import pandas as pd
import os
import execjs
from fake_useragent import UserAgent
import json
from scrapy.http import Request
from scrapyfunds.items import ScrapyfundsItem
from scrapy_splash import SplashRequest,SplashFormRequest


class FundsSpider(scrapy.Spider):
    # 抓取基金评级，季度级
    name = 'FundsRating'  # 唯一，用于区别Spider。运行爬虫时，就要使用该名字
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
        if os.path.exists('./data/fund/fund_rating_pd.csv'):
            os.remove('./data/fund/fund_rating_pd.csv')
        # SplashRequest对象，前两个参数依然是请求的URL和回调函数。另外我们还可以
        # 通过args传递一些渲染参数，例如等待时间wait等，还可以根据endpoint参数指定渲
        # 染接口。更多参数可以参考文档说明：https://github.com/scrapy-plugins/scrapy-
        # splash#requests。

        url = 'http://fund.eastmoney.com/data/fundrating.html'
        yield SplashRequest(
            url=url,
            callback=self.parse_funds_list,
            meta={'title': 'xxxx'},
            args={
                'wait': 1,
            }
        )

        # requests = []
        #
        # request = scrapy.Request(url, callback=self.parse_funds_list)
        # requests.append(request)
        #
        # return requests

    def parse_funds_list(self, response):
        # print("response.status", response.status)
        # print("response.url", response.url)
        # print("response.meta", response.meta)
        # datas = response.xpath('//div[@id="fundinfo"]')
        data = response.xpath('//table[@class="table0"]')
        # print("data", len(data))
        # data_thead_r = data.xpath('thead/tr/th/text()').getall()
        data_thead_th = data.xpath('thead/tr/th')
        columns_list = []
        for th in data_thead_th:
            c_name = ''
            if th.xpath('span/text()').get() is not None:
                c_name = th.xpath('span/text()').get()
            elif th.xpath('text()').get() is not None:
                c_name = th.xpath('text()').get()
            # print("c_name", c_name)
            if c_name == '代码':
                c_name = 'code'
            if c_name != '基金公司' and c_name != '相关链接' and c_name != '简称' and c_name != '基金经理' and c_name != '操作':
                columns_list.append(c_name)
        # print("columns_list", len(columns_list), columns_list)
        fund_rating_pd = pd.DataFrame(columns=columns_list)
        # print("----------")
        trs = data.xpath('tbody/tr')
        # tr = trs[0]
        # print(trs.getall())
        for tr in trs:
            td_list = []
            tds = tr.xpath('td')
            # print("tds", len(tds.getall()), tds.getall())
            td_list.append(tds[0].xpath('a/text()').get())
            td_list.append(tds[5].xpath('text()').get().split('家')[0])
            # print("tds[6].xpath('text()').get()", tds[6].xpath('text()').get())
            if tds[6].xpath('text()').get() is not None:
                td_list.append(len(tds[6].xpath('text()').get().split('★')) - 1)
            else:
                td_list.append(0)
            if tds[7].xpath('text()').get() is not None:
                td_list.append(len(tds[7].xpath('text()').get().split('★')) - 1)
            else:
                td_list.append(0)
            if tds[8].xpath('text()').get() is not None:
                td_list.append(len(tds[8].xpath('text()').get().split('★')) - 1)
            else:
                td_list.append(0)

            td_list.append(tds[9].xpath('a/text()').get())
            td_pd = pd.DataFrame([td_list], columns=columns_list)
            fund_rating_pd = fund_rating_pd.append(td_pd, ignore_index=True)
        print("fund_rating_pd", fund_rating_pd.shape)

        fundsItems = []
        fundsItem = ScrapyfundsItem()
        fundsItem['fund_rating_pd'] = fund_rating_pd
        fundsItems.append(fundsItem)
        return fundsItems
