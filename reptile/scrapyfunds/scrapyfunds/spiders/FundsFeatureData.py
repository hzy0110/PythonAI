import scrapy
import pandas as pd
import os
from scrapyfunds.items import ScrapyfundsItem
import numpy as np
import arrow
import time


class FundsSpider(scrapy.Spider):
    # 抓取基金特色数据，季度级
    name = 'FundsFeatureData'  # 唯一，用于区别Spider。运行爬虫时，就要使用该名字
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
            os.remove('./data/fund/feature_data_narrow_pd.csv')

        url = 'http://fundf10.eastmoney.com/tsdata_{}.html'
        requests_list = []
        # Test
        # url_test = 'http://fundf10.eastmoney.com/tsdata_000001.html'
        # url_test = 'http://fundf10.eastmoney.com/tsdata_040001.html'
        url_test = 'http://fundf10.eastmoney.com/tsdata_000002.html'
        # url_test = 'http://fundf10.eastmoney.com/tsdata_000003.html'
        # url_test = 'http://fundf10.eastmoney.com/tsdata_000006.html'
        request = scrapy.Request(url_test, callback=self.parse_funds_list, meta={'fund_code': '040001'})
        requests_list.append(request)

        # allCode = np.loadtxt("./data/fund/code_np.csv", dtype=np.str, delimiter=',').tolist()
        # for fund_code in allCode:
        #     request = scrapy.Request(url.format(fund_code), callback=self.parse_funds_list, meta={'fund_code': fund_code})
        #     requests_list.append(request)

        return requests_list

    def parse_funds_list(self, response):
        # print(dir(response))
        if response.status == 200:
            fund_code = response.meta['fund_code']
            # print('fund_code', fund_code)
            fundsItems = []
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

            data_1 = response.xpath('//table[@class="fxtb"]')
            data_2 = response.xpath('//table[@class="fgtb"]')

            data_tr = data_1.xpath('tr')
            y1 = data_tr[0].xpath('th/text()').getall()[1]
            y2 = data_tr[0].xpath('th/text()').getall()[2]
            y3 = data_tr[0].xpath('th/text()').getall()[3]
            y1 = '_1y'
            y2 = '_2y'
            y3 = '_3y'

            norm1 = data_tr[1].xpath('td/text()').getall()[0]
            near = '_near'
            if norm1 == '标准差':
                norm1 = 'std'
            norm2 = data_tr[2].xpath('td/text()').getall()[0]
            if norm2 == '夏普比率':
                norm2 = 'sharp'
            v1 = data_tr[1].xpath('td/text()').getall()[1].replace('%', '')
            v2 = data_tr[1].xpath('td/text()').getall()[2].replace('%', '')
            v3 = data_tr[1].xpath('td/text()').getall()[3].replace('%', '')
            if v1 != '--':
                v1_int = float(v1) / 100
            else:
                v1_int = 0
            if v2 != '--':
                v2_int = float(v2) / 100
            else:
                v2_int = 0
            if v3 != '--':
                v3_int = float(v3) / 100
            else:
                v3_int = 0
            v4 = data_tr[2].xpath('td/text()').getall()[1]
            v5 = data_tr[2].xpath('td/text()').getall()[2]
            v6 = data_tr[2].xpath('td/text()').getall()[3]
            if v4 != '--':
                v4_int = float(v4)
            else:
                v4_int = 0
            if v5 != '--':
                v5_int = float(v5)
            else:
                v5_int = 0
            if v6 != '--':
                v6_int = float(v6)
            else:
                v6_int = 0

            columns = [norm1 + y1 + near, norm1 + y2 + near, norm1 + y3 + near, norm2 + y1 + near, norm2 + y2 + near,
                       norm2 + y3 + near]

            fund_feature_data_pd = pd.DataFrame([[v1_int, v2_int, v3_int, v4_int, v5_int, v6_int]], columns=columns)

            data_tr = data_2.xpath('tr')
            # print('len(data_tr)', len(data_tr))
            if len(data_tr) == 9:
                name = data_tr[0].xpath('th/text()').getall()[1]
                columns_list = []
                values_list = []
                for tr in data_tr:
                    # qname = tr.xpath('td/text()').getall()[0]
                    if len(tr.xpath('td/text()').getall()) > 0:
                        quarter_name = tr.xpath('td/text()').getall()[0]
                        values = tr.xpath('td/text()').getall()[1]
                        # columns_list.append(name + quarter_name)
                        columns_list.append(quarter_name)
                        values_list.append(self.fund_invest_style(values))
                invest_style_pd = pd.DataFrame([values_list], columns=['invest_style_1q_near', 'invest_style_2q_near',
                                                                       'invest_style_3q_near',
                                                                       'invest_style_4q_near', 'invest_style_5q_near',
                                                                       'invest_style_6q_near',
                                                                       'invest_style_7q_near', 'invest_style_8q_near'])
                # print('invest_style_pd', invest_style_pd.T)

            elif len(data_tr) > 0:
                name = data_tr[0].xpath('th/text()').getall()[1]
                columns_list = []
                values_np = np.array(['', '', '', '', '', '', '', ''], dtype=np.object)
                # for tr in data_tr:
                for i in range(len(data_tr)):

                    # qname = tr.xpath('td/text()').getall()[0]
                    if len(data_tr[i].xpath('td/text()').getall()) > 0:
                        quarter_name = data_tr[i].xpath('td/text()').getall()[0]
                        values = data_tr[i].xpath('td/text()').getall()[1]
                        columns_list.append(quarter_name)
                        values_np[i - 1] = self.fund_invest_style(values)
                # print('columns_list', columns_list)
                invest_style_pd = pd.DataFrame([values_np], columns=['invest_style_1q_near', 'invest_style_2q_near',
                                                                     'invest_style_3q_near',
                                                                     'invest_style_4q_near', 'invest_style_5q_near',
                                                                     'invest_style_6q_near',
                                                                     'invest_style_7q_near', 'invest_style_8q_near'])
                # print('invest_style_pd', invest_style_pd.T)
            else:
                invest_style_pd = pd.DataFrame([], columns=['invest_style_1q_near', 'invest_style_2q_near',
                                                            'invest_style_3q_near',
                                                            'invest_style_4q_near', 'invest_style_5q_near',
                                                            'invest_style_6q_near',
                                                            'invest_style_7q_near', 'invest_style_8q_near'])

            # 合并 DataFrame
            # feature_data_pd = pd.concat([create_date_pd, fund_feature_data_pd], axis=1)
            # feature_data_pd = pd.concat([feature_data_pd, invest_style_pd], axis=1)
            feature_data_pd = invest_style_pd
            # print("feature_data_pd", feature_data_pd, feature_data_pd.info())

            # 处理 history 用的表
            feature_data_narrow_pd = invest_style_pd
            # feature_data_narrow_pd.columns = self.getQuarterDateByStr()
            feature_data_narrow_pd = feature_data_narrow_pd.T

            feature_data_narrow_pd.insert(0, 'fund_code', fund_code)
            feature_data_narrow_pd.insert(1, 'trade_timestamp', self.getQuarterDateByStr()[1])
            feature_data_narrow_pd.insert(2, 'trade_date', self.getQuarterDateByStr()[0])

            # print('feature_data_narrow_pd', feature_data_narrow_pd.shape)
            if feature_data_narrow_pd.shape[1] == 4:
                feature_data_narrow_pd.columns = ['fund_code', 'trade_timestamp', 'trade_date', 'invest_style']
            else:
                feature_data_narrow_pd.insert(3, 'invest_style', [0, 0, 0, 0, 0, 0, 0, 0])
            feature_data_narrow_pd.fillna(0, inplace=True)
            # print('feature_data_narrow_pd', feature_data_narrow_pd)
            # print('feature_data_narrow_pd', feature_data_narrow_pd)
            # print('feature_data_pd', feature_data_pd.T)

        fundsItem = ScrapyfundsItem()
        fundsItem['feature_data_pd'] = feature_data_pd
        fundsItem['feature_data_narrow_pd'] = feature_data_narrow_pd
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

    def timestr_to_13timestamp(self, dt):
        timearray = time.strptime(dt, '%Y-%m-%d')
        timestamp13 = int(time.mktime(timearray))
        return int(round(timestamp13 * 1000))
        # year = quarterStr.replace('基金投资风格', '').split('年')[0] + '-'
        # quarter = quarterStr.replace('基金投资风格', '').split('年')[1].split('季度')[0]
        # quarterDate = ''
        # if quarter == "1":
        #     quarterDate = '3-31'
        # elif quarter == "2":
        #     quarterDate = '6-30'
        # elif quarter == "3":
        #     quarterDate = '9-30'
        # elif quarter == "4":
        #     quarterDate = '12-31'
        # styleStr = year + quarterDate
        # styleDate = datetime.datetime.strptime(styleStr, '%Y-%m-%d')
        # return styleDate.strftime("%Y-%m-%d")

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



    # def get_quarter_end():
    #     now = arrow.utcnow().to("local")
    #     return now.ceil("quarter")
