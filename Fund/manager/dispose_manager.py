from tools.reptile import Reptile
from dao.dao import Dao
import time

class Manager:
    def __init__(self):
        pass

    def run_menager(self):
        start = time.time()
        manager_info_pd, manager_current_pd = Reptile().get_all_manage()
        print('爬取处理结束，开始写库')
        Dao().save_manager_info(manager_info_pd)
        Dao().save_manager_current(manager_current_pd)
        print('爬取处理结束，写库结束')
        end = time.time() - start
        print('用时1', end)
