import scrapy
import pandas as pd
import os
from scrapyfunds.items import ScrapyfundsItem
import numpy as np
import arrow
import time
# from scrapy_splash import SplashRequest

class FundsSpider(scrapy.Spider):
    # 抓取基金特色数据，季度级
    name = 'FundsBaseInfo'  # 唯一，用于区别Spider。运行爬虫时，就要使用该名字
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
        if os.path.exists('./data/fund/fund_base_info_pd.csv'):
            os.remove('./data/fund/fund_base_info_pd.csv')

        url = 'http://fundf10.eastmoney.com/jbgk_{}.html'
        requests_list = []
        # Test
        # # url_test = 'http://fundf10.eastmoney.com/jbgk_000001.html'
        # # url_test = 'http://fundf10.eastmoney.com/jbgk_040001.html'
        # # url_test = 'http://fundf10.eastmoney.com/jbgk_000006.html'
        # # url_test = 'http://fundf10.eastmoney.com/jbgk_000004.html'
        # url_test = 'http://fundf10.eastmoney.com/jbgk_002953.html'
        # request = scrapy.Request(url_test, callback=self.parse_funds_list, meta={'fund_code': '000002'})
        # requests_list.append(request)

        allCode = np.loadtxt("./data/fund/code_np.csv", dtype=np.str, delimiter=',').tolist()
        for fund_code in allCode:
            request = scrapy.Request(url.format(fund_code), callback=self.parse_funds_list, meta={'fund_code': fund_code})
            requests_list.append(request)
        return requests_list

    def parse_funds_list(self, response):
        # print(dir(response))

        if response.status == 200:
            fund_code = response.meta['fund_code']
            # fund_base_info_pd = pd.DataFrame([fund_code], columns=['fund_code'])
            tr = response.xpath('//table[@class="info w790"]/tr')
            fund_name = tr[0].xpath('td')[1].xpath('text()').get()
            # print('fund_name', fund_name)
            fund_type_name = tr[1].xpath('td')[1].xpath('text()').get()
            fund_type = self.fund_type_2_num(fund_type_name)
            # print('fund_type_name', fund_type_name)
            # print('fund_type', fund_type)
            # print('fund_code', fund_code)
            cd = tr[2].xpath('td')[1].xpath('text()').get().replace(" ", "").split('/')[0]
            if cd != '' and cd is not None:
                fund_create_date = time.strptime(cd, '%Y年%m月%d日')
                fund_create_date = time.strftime("%Y-%m-%d", fund_create_date)
                create_timestamp = self.timestr_to_13timestamp(fund_create_date)
            else:
                create_timestamp = 0
            # print('fund_create_date', fund_create_date)
            create_share_scale = tr[2].xpath('td')[1].xpath('text()').get().replace(
                " ", "").split('/')[1].replace("亿份", "")
            if create_share_scale == '--':
                create_share_scale = 0
            # print('create_share_scale', create_share_scale)
            current_asset_scale = tr[3].xpath('td')[0].xpath('text()').get().replace(
                " ", "").split('亿')[0]
            current_asset_scale = current_asset_scale.replace(',', '')
            if current_asset_scale == '---':
                current_asset_scale = 0
            # print('current_asset_scale', current_asset_scale)
            if tr[3].xpath('td')[1].xpath('a/text()').get() is not None:
                current_share_scale = tr[3].xpath('td')[1].xpath('a/text()').get().replace(
                    " ", "").split('亿')[0]
            else:
                current_share_scale = 0
            # print('current_share_scale', current_share_scale)
            company_code = tr[4].xpath('td')[0].css("a::attr(href)").getall()[-1].replace(
                "http://fund.eastmoney.com/company/", "").replace(".html", "")
            # print('company_code', company_code)
            established_dividend_amount = tr[5].xpath('td')[1].xpath('a/text()').get().replace(
                " ", "").replace('每份累计', '').replace('(', '').replace(')', '').replace('次', '').split('元')[0]
            established_dividend_count = tr[5].xpath('td')[1].xpath('a/text()').get().replace(
                " ", "").replace('每份累计', '').replace('（', '').replace('）', '').replace('次', '').split('元')[1]
            # print('established_dividend_amount', established_dividend_amount)
            # print('established_dividend_count', established_dividend_count)

            title = tr[8].xpath('th/text()')[0].get()
            if title == '最高申购费率':
                original_rate = tr[8].xpath('td')[0].xpath('text()').get().split('（')[0].replace('%', '')
                current_rate = original_rate
            else:
                td = response.xpath('//table[@class="info w790"]/td')
                # print('original_rate', td.getall())
                original_rate = td[0].xpath('span/text()').get().split('（')[0].replace('%', '')
                current_rate = td[0].xpath('span/span/text()').get().split('（')[0].replace('%', '')
            if original_rate == '---':
                original_rate = 0.00
            if current_rate == '---':
                current_rate = 0.00
            if response.xpath('//a[@class="btn  btn-red "]/span/text()').getall():
                min_buy = response.xpath('//a[@class="btn  btn-red "]/span/text()').getall()[1].split('元')[0]
                min_buy.replace('万', '0000')
            else:
                min_buy = 0
            fund_base_info_pd = pd.DataFrame([[fund_code, fund_name, fund_type, fund_type_name, min_buy,
                                               create_share_scale, current_asset_scale, current_share_scale,
                                               established_dividend_amount, established_dividend_count,
                                                original_rate, current_rate, company_code,
                                                create_timestamp]],
              columns=['fund_code', 'fund_name', 'fund_type', 'fund_type_name', 'min_buy',
                       'create_share_scale', 'current_asset_scale', 'current_share_scale',
                       'established_dividend_amount', 'established_dividend_count',
                       'original_rate', 'current_rate', 'company_code', 'fund_create_timestamp'])

            # print(fund_base_info_pd.T)
            # fundsItems = []
            # # 基金名称
            # fund_name = response.xpath('//div[@class="col-left"]/h4/a/text()').get().split(' ')[0]
            # # 成立日期
            # create_date = response.xpath('//div[@class="bs_gl"]/p/label/span/text()').get()
            # # 类型
            # fund_type_name = response.xpath('//div[@class="bs_gl"]/p/label/span/text()').getall()[1]
            # fund_type = self.fund_type_2_num(fund_type_name)
            # # 最低购买
            # min_buy = response.xpath('//a[@class="btn  btn-red "]/span/text()').getall()[1].split('元')[0]
            # # 购买手续费
            # rate = response.xpath('//div[@class="col-right"]/p')
            # original_rate = rate[2].xpath('label/b/text()').getall()[0].replace('%', '')
            # current_rate = rate[2].xpath('label/b/text()').getall()[1].replace('%', '')
            # print(original_rate)
            # print(current_rate)
            # print(create_date)
            # if create_date is None:
            #     return fundsItems
            # if create_date == '---':
            #     create_date = '无日期'
            #
            # if create_date != '无日期':
            #     create_timestamp = self.timestr_to_13timestamp(create_date)
            # else:
            #     create_timestamp = 0

            # # 公司代码
            # company_code = response.xpath('//div[@class="bs_gl"]/p/label').css("a::attr(href)").getall()[-1].replace(
            #     "http://fund.eastmoney.com/company/", "").replace(".html", "")
            # company_name = response.xpath('//div[@class="bs_gl"]/p/label/a/text()').getall()[-1]

            # if create_date != '无日期':
            #     create_date_pd = pd.DataFrame([[fund_code, fund_name, fund_type, fund_type_name, min_buy,
            #                                     original_rate, current_rate, company_code, create_date, create_timestamp]],
            #                                   columns=['fund_code', 'fund_name', 'fund_type', 'fund_type_name', 'min_buy',
            #                                            'original_rate', 'current_rate', 'company_code', 'fund_create_date', 'fund_create_timestamp'])
            #
            # else:
            # create_date_pd = pd.DataFrame([[fund_code, fund_name, fund_type, fund_type_name, min_buy,
            #                                 original_rate, current_rate, company_code, create_timestamp]],
            #                               columns=['fund_code', 'fund_name', 'fund_type', 'fund_type_name', 'min_buy',
            #                                        'original_rate', 'current_rate', 'company_code', 'fund_create_timestamp'])

        fundsItems = []
        fundsItem = ScrapyfundsItem()
        fundsItem['fund_base_info'] = fund_base_info_pd
        fundsItems.append(fundsItem)
        return fundsItems

    def getQuarterDateByStr(self):
        quarter_list = []
        quarter_ts_list = []
        for i in range(-1, -9, -1):
            # print('i', i)
            a = arrow.utcnow().to("local").shift(months=i * 3)
            a = a.ceil("quarter")
            a_str = a.format('YYYY-MM-DD')
            # a_str2ts = a.format('YYYY-MM-DD')
            ts = self.timestr_to_13timestamp(a_str)
            quarter_ts_list.append(ts)
            # print(a)
            # print(a.ceil("quarter").format('YYYY-MM-DD'))
            quarter_list.append(a_str)
        return quarter_list, quarter_ts_list

    def fund_invest_style(self, style_name):
        if style_name == '大盘价值':
            return 1
        elif style_name == '大盘平衡':
            return 2
        elif style_name == '大盘成长':
            return 3
        elif style_name == '中盘价值':
            return 4
        elif style_name == '中盘平衡':
            return 5
        elif style_name == '中盘成长':
            return 6
        elif style_name == '小盘价值':
            return 7
        elif style_name == '小盘平衡':
            return 8
        elif style_name == '小盘成长':
            return 9

    def getTimedelta(self, dayStr):
        if dayStr == '无日期':
            return 0
        now = arrow.now()
        target_day = arrow.get(dayStr)
        diff = now - target_day
        return diff.days

    # 获取时间戳和时间戳列表
    def timestr_to_13timestamp(self, dt):
        timearray = time.strptime(dt, '%Y-%m-%d')
        timestamp13 = int(time.mktime(timearray))
        return int(round(timestamp13 * 1000))

    # 获取基金类型的 typeid
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
