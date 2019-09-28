from sqlalchemy import create_engine
import os

# 定义编码格式
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'


class Conn:
    def __init__(self):
        self.connModel = "mysql+pymysql"

    def getEngineFunds(self):
        return create_engine(self.connModel + "://hzy:&807610Mysql@127.0.0.1:3306/fund?charset=utf8", max_overflow=5)
