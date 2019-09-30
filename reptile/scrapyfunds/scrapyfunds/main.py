from scrapy import cmdline
# cmdline.execute("scrapy crawl FundsGrandTotalReportPer -s LOG_FILE=./data/log/FundsGrandTotalReportPer.log".split())
# cmdline.execute("scrapy crawl FundsZSZQRating -s LOG_FILE=./data/log/FundsZSZQRating.log".split())
# cmdline.execute("scrapy crawl FundsSHZQRating -s LOG_FILE=./data/log/FundsSHZQRating.log".split())
# cmdline.execute("scrapy crawl FundsJAJXRating -s LOG_FILE=./data/log/FundsJAJXRating.log".split())
# cmdline.execute("scrapy crawl FundsFeatureData -s LOG_FILE=./data/log/FundsManagerFeatureData.log".split())
cmdline.execute("scrapy crawl FundsBaseInfo".split())
# cmdline.execute("scrapy crawl FundsBaseInfo -s LOG_FILE=./data/log/FundsBaseInfo.log".split())
# cmdline.execute("scrapy crawl FundsRating -s LOG_FILE=./data/log/FundsRating.log".split())
# cmdline.execute("scrapy crawl FundsManagerSimilar -s LOG_FILE=./data/log/FundsManagerSimilar.log".split())
# cmdline.execute("scrapy crawl ManagerCurrentFund -s LOG_FILE=./data/log/ManagerCurrentFund.log".split())
# 基金评级 http://fundf10.eastmoney.com/jjpj_040001.html
# 基金持仓 http://fundf10.eastmoney.com/ccmx_040001.html
# 债券持仓 http://fundf10.eastmoney.com/ccmx1_040001.html
# 持仓变动走势 http://fundf10.eastmoney.com/ccbdzs_000003.html
# 行业配置 http://fundf10.eastmoney.com/hybd_040001_C_new.html
# 资产配置 http://fundf10.eastmoney.com/zcpz_000003.html
# 重大变动 http://fundf10.eastmoney.com/ccbd_000003.html
# 规模变动 http://fundf10.eastmoney.com/gmbd_000003.html
# 持有人结构 http://fundf10.eastmoney.com/cyrjg_000003.html
