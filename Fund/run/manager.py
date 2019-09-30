from reptile.manager import ReptileManager
from calculation.manager import CalManager
from dao.dao import Dao
import time
import pandas as pd

class Manager:
    def __init__(self):
        pass

    def menager_info_all_manager(self):
        start = time.time()
        manager_info_pd, manager_current_pd = ReptileManager().get_all_manager()
        print('爬取处理结束，开始写库')
        Dao().save_manager_info(manager_info_pd)
        Dao().save_manager_current(manager_current_pd)
        print('爬取处理结束，写库结束')
        end = time.time() - start
        print('用时1', end)

    def manager_current_by_manager_page(self):
        start = time.time()
        manager_current_pd = pd.read_csv('../data/manager/manager_current_pd.csv', low_memory=False)
        Dao().save_manager_current(manager_current_pd)
        end = time.time() - start
        print('用时', end)

    def manager_history_by_fund_manager_page(self):
        start = time.time()
        manager_history_similar_pd = pd.read_csv('../data/manager/manager_history_similar_pd.csv', low_memory=False)
        Dao().save_manager_history(manager_history_similar_pd)
        end = time.time() - start
        print('用时', end)

    def manager_get_good_at_type(self):
        rowcount, is_insert = Dao().get_manager_good_at_type()
        print("update rowcount:", rowcount)

