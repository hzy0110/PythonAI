from fund.dao.dao import Dao
import pandas as pd
import numpy as np


class FundInfo:
    def __init__(self):
        self.a = 'a'

    def insert_all_code(self):
        allCode = np.loadtxt("./data/fund/code_np.csv", dtype=np.str, delimiter=',').tolist()
        for code in allCode:
            Dao().insert_fund_code(code)

    def update_fund_by_code(self, fund_code):
        pass

