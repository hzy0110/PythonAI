import scrapy
import json
from scrapy.http import Request
from scrapyfunds.items import ScrapyfundsItem


class FundsSpider(scrapy.Spider):
    # Demo
    name = 'fundsList'  # 唯一，用于区别Spider。运行爬虫时，就要使用该名字
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
        url = 'https://fundapi.eastmoney.com/fundtradenew.aspx?ft=pg&sc=1n&st=desc&pi=1&pn=3000&cp=&ct=&cd=&ms=&fr=&plevel=&fst=&ftype=&fr1=&fl=0&isab='
        requests = []
        request = scrapy.Request(url, callback=self.parse_funds_list)
        requests.append(request)
        return requests

    def parse_funds_list(self, response):
        datas = response.body.decode('UTF-8')

        # 取出json部门
        datas = datas[datas.find('{'):datas.find('}') + 1]  # 从出现第一个{开始，取到}

        # 给json各字段名添加双引号
        datas = datas.replace('datas', '\"datas\"')
        datas = datas.replace('allRecords', '\"allRecords\"')
        datas = datas.replace('pageIndex', '\"pageIndex\"')
        datas = datas.replace('pageNum', '\"pageNum\"')
        datas = datas.replace('allPages', '\"allPages\"')

        jsonBody = json.loads(datas)
        jsonDatas = jsonBody['datas']

        fundsItems = []
        for data in jsonDatas:
            fundsItem = ScrapyfundsItem()
            fundsArray = data.split('|')
            fundsItem['code'] = fundsArray[0]
            fundsItem['name'] = fundsArray[1]
            fundsItem['day'] = fundsArray[3]
            fundsItem['unitNetWorth'] = fundsArray[4]
            fundsItem['dayOfGrowth'] = fundsArray[5]
            fundsItem['recent1Week'] = fundsArray[6]
            fundsItem['recent1Month'] = fundsArray[7]
            fundsItem['recent3Month'] = fundsArray[8]
            fundsItem['recent6Month'] = fundsArray[9]
            fundsItem['recent1Year'] = fundsArray[10]
            fundsItem['recent2Year'] = fundsArray[11]
            fundsItem['recent3Year'] = fundsArray[12]
            fundsItem['fromThisYear'] = fundsArray[13]
            fundsItem['fromBuild'] = fundsArray[14]
            fundsItem['serviceCharge'] = fundsArray[18]
            fundsItem['upEnoughAmount'] = fundsArray[24]
            fundsItems.append(fundsItem)
        return fundsItems
