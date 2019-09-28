from fund.tools.reptile import Reptile
from fund.dao.dao import Dao
import pandas as pd


class Manager:
    def __init__(self):
        pass

    def run_menager(self):
        manager_info_pd, manager_current_pd = Reptile().get_all_manage()
        Dao().save_manager_info(manager_info_pd)
        Dao().save_manager_current(manager_current_pd)
