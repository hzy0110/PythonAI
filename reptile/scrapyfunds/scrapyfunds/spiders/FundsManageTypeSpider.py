import scrapy
import pandas as pd
import os
from fake_useragent import UserAgent
import json
from scrapy.http import Request
from scrapyfunds.items import ScrapyfundsItem


class FundsSpider(scrapy.Spider):
    # 抓取基金经理历任和现任的基金信息
    name = 'fundsManageTypeList'  # 唯一，用于区别Spider。运行爬虫时，就要使用该名字
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
        url = 'http://fund.eastmoney.com/manager/{}.html'

        if os.path.exists('./data/manager/manager_history_pd.csv'):
            os.remove('./data/manager/manager_history_pd.csv')
        if os.path.exists('./data/manager/manager_current_pd.csv'):
            os.remove('./data/manager/manager_current_pd.csv')
        requests = []

        # Test
        # url_test = 'http://fund.eastmoney.com/manager/30395964.html'
        # request = scrapy.Request(url_test, callback=self.parse_funds_list, meta={'managerID': '30395964'})
        # requests.append(request)


        allManager_pd = pd.read_csv('/Users/hzy/code/PycharmProjects/reptile/data/manager/allManager_pd.csv', low_memory=False)
        allManager_id_s = allManager_pd['manageID']

        for manage_id in allManager_id_s:
            request = scrapy.Request(url.format(manage_id), callback=self.parse_funds_list, meta={'managerID': manage_id})
            requests.append(request)
        return requests

    def parse_funds_list(self, response):
        manager_id = response.meta['managerID']
        datas = response.xpath('//table[@class="ftrs"]')
        manages_list = []
        for data in datas:
            data_thead_r = data.xpath('thead/tr/th/text()').getall()
            manage_pd = pd.DataFrame(columns=data_thead_r)
            data_tbody = data.xpath('tbody/tr')
            for body_tr in data_tbody:
                tds = body_tr.xpath('td')
                manage_body_list = []
                for td in tds:
                    # print('d2', len(d2.getall()), type(d2))
                    if td.xpath('text()').get() is not None and td.xpath('span/text()').get():
                        st = td.xpath('span/text()').getall()
                        st.append(td.xpath('text()').get())
                        manage_body_list.append(''.join(st))
                        # print("st", ''.join(st))

                    elif td.xpath('text()').get() is not None:
                        t = td.xpath('text()').get()
                        manage_body_list.append(t)
                        # print('t', t)
                    elif td.xpath('span/text()').get() is not None:
                        if len(td.xpath('span/text()').getall()) == 1:
                            span = td.xpath('span/text()').get()
                            # print("span", span)
                        else:
                            span = td.xpath('span/text()').getall()
                            # print('span', span)
                        manage_body_list.append(span)
                    elif td.xpath('a/text()').get() is not None:
                        if len(td.xpath('a/text()').getall()) == 1:
                            a = td.xpath('a/text()').get()
                            # print("a", a)
                        else:
                            a = td.xpath('a/text()').getall()
                            # print("a", ','.join(a))
                        manage_body_list.append(a)

                # print("manage_body_list", manage_body_list)
                manage_body_pd = pd.DataFrame([manage_body_list], columns=data_thead_r)
                manage_pd = manage_pd.append(manage_body_pd, ignore_index=True)
            # print(manage_pd)
            manage_pd.insert(0, '经理ID', manager_id)
            manages_list.append(manage_pd)
        # 对于Selector类型的对象，并不能使用extract_first()方法，而使用get()可以

        fundsItems = []
        fundsItem = ScrapyfundsItem()
        fundsItem['manager_pd'] = manages_list
        # fundsItem['manager_id'] = manager_id
        fundsItems.append(fundsItem)
        return fundsItems
