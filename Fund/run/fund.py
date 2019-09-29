from reptile.fund import ReptileFund
from tools.tool import Tool
from dao.dao import Dao
import pandas as pd
import time
import sys
from multiprocessing import Process, Queue


class Fund:
    def __init__(self):
        pass

    def fund_info_pzdata(self, allCode, allType, error_queue):
        for i in range(len(allCode)):
            if i % 200 == 0:
                print('进度:', i)
            try:
                code, ishb, basic_info_pd, funds = ReptileFund().get_pingzhongdata(allCode[i], allType[i])
                if ishb is not None:
                    if ishb:
                        fund_wide_pd, rateInSimilar_pd, grandTotal_pd, worth_pd, current_fund_manager_pd = Tool().get_hb_pingzhongdata_2_df(
                            basic_info_pd, funds)
                    else:
                        fund_wide_pd, rateInSimilar_pd, grandTotal_pd, worth_pd, current_fund_manager_pd = Tool().get_pingzhongdata_2_df(
                            basic_info_pd, funds)
                    # print(code, '用时2', end)
                    Dao().save_fund_info(fund_wide_pd, 'fund_code', int(code))
                    # print(code, '用时3', end)

                    if not rateInSimilar_pd.empty:
                        Dao().save_fund_history(rateInSimilar_pd, 'fund_code', int(code))
                        pass
                    if not grandTotal_pd.empty:
                        Dao().save_fund_history(grandTotal_pd, 'fund_code', int(code))
                        pass
                    if not worth_pd.empty:
                        Dao().save_fund_history(worth_pd, 'fund_code', int(code))
                        pass
                    if not current_fund_manager_pd.empty:
                        Dao().save_manager_info(current_fund_manager_pd)
                    error_queue.put(None)
            except Exception as e:
                print("错误 code", code)
                error_queue.put(e, code)

                # sys.stdout.flush()
        print("end")

    def fund_info_feature(self):
        fund_info_feature_pd = pd.read_csv('../data/fund/feature_data_pd.csv', low_memory=False)
        rowcount, is_insert = Dao().save_fund_info(fund_info_feature_pd)
        print('rowcount', rowcount)

    def fund_history_feature(self):
        pass

    def single_fund_info(self, code_s, type_s):
        code, ishb, basic_info_pd, funds = ReptileFund().get_pingzhongdata(code_s, type_s)
        if ishb is not None:
            if ishb:
                fund_wide_pd, rateInSimilar_pd, grandTotal_pd, worth_pd, current_fund_manager_pd = Tool().get_hb_pingzhongdata_2_df(
                    basic_info_pd, funds)
            else:
                fund_wide_pd, rateInSimilar_pd, grandTotal_pd, worth_pd, current_fund_manager_pd = Tool().get_pingzhongdata_2_df(
                    basic_info_pd, funds)
            # print(code, '用时2', end)
            Dao().save_fund_info(fund_wide_pd, 'fund_code', int(code))
            # print(code, '用时3', end)

            if not rateInSimilar_pd.empty:
                Dao().save_fund_history(rateInSimilar_pd, 'fund_code', int(code))
                pass
            if not grandTotal_pd.empty:
                Dao().save_fund_history(grandTotal_pd, 'fund_code', int(code))
                pass
            if not worth_pd.empty:
                Dao().save_fund_history(worth_pd, 'fund_code', int(code))
                pass
            if not current_fund_manager_pd.empty:
                Dao().save_manager_info(current_fund_manager_pd)

            # sys.stdout.flush()

    def run_reptile_pinzhongdata_multiple(self):
        allCodeType = ReptileFund().get_all_code_type()
        error_queue = Queue()  # 实现子进程和主进程之间的报错信息通信
        # allCode1 = allCodeType.loc[0:2, 'code'].values.tolist()
        # allType1 = allCodeType.loc[0:2, 'type'].values.tolist()
        # allCode2 = allCodeType.loc[2:4, 'code'].values.tolist()
        # allType2 = allCodeType.loc[2:4, 'type'].values.tolist()
        # allCode3 = allCodeType.loc[4:6, 'code'].values.tolist()
        # allType3 = allCodeType.loc[4:6, 'type'].values.tolist()
        # allCode4 = allCodeType.loc[6:8, 'code'].values.tolist()
        # allType4 = allCodeType.loc[6:8, 'type'].values.tolist()
        # allCode5 = allCodeType.loc[8:10, 'code'].values.tolist()
        # allType5 = allCodeType.loc[8:10, 'type'].values.tolist()
        # allCode6 = allCodeType.loc[10:12, 'code'].values.tolist()
        # allType6 = allCodeType.loc[10:12, 'type'].values.tolist()
        # allCode7 = allCodeType.loc[12:14, 'code'].values.tolist()
        # allType7 = allCodeType.loc[12:14, 'type'].values.tolist()
        # allCode8 = allCodeType.loc[14:16, 'code'].values.tolist()
        # allType8 = allCodeType.loc[14:16, 'type'].values.tolist()

        allCode1 = allCodeType.loc[0:1000, 'code'].values.tolist()
        allType1 = allCodeType.loc[0:1000, 'type'].values.tolist()
        allCode2 = allCodeType.loc[1000:2000, 'code'].values.tolist()
        allType2 = allCodeType.loc[1000:2000, 'type'].values.tolist()
        allCode3 = allCodeType.loc[2000:3000, 'code'].values.tolist()
        allType3 = allCodeType.loc[2000:3000, 'type'].values.tolist()
        allCode4 = allCodeType.loc[3000:4000, 'code'].values.tolist()
        allType4 = allCodeType.loc[3000:4000, 'type'].values.tolist()
        allCode5 = allCodeType.loc[4000:5000, 'code'].values.tolist()
        allType5 = allCodeType.loc[4000:5000, 'type'].values.tolist()
        allCode6 = allCodeType.loc[6000:7000, 'code'].values.tolist()
        allType6 = allCodeType.loc[6000:7000, 'type'].values.tolist()
        allCode7 = allCodeType.loc[7000:8000, 'code'].values.tolist()
        allType7 = allCodeType.loc[7000:8000, 'type'].values.tolist()
        allCode8 = allCodeType.loc[8000:, 'code'].values.tolist()
        allType8 = allCodeType.loc[8000:, 'type'].values.tolist()

        start = time.time()

        p1 = Process(target=self.fund_info_pzdata, args=(allCode1, allType1, error_queue))
        p2 = Process(target=self.fund_info_pzdata, args=(allCode2, allType2, error_queue))
        p3 = Process(target=self.fund_info_pzdata, args=(allCode3, allType3, error_queue))
        p4 = Process(target=self.fund_info_pzdata, args=(allCode4, allType4, error_queue))
        p5 = Process(target=self.fund_info_pzdata, args=(allCode5, allType5, error_queue))
        p6 = Process(target=self.fund_info_pzdata, args=(allCode6, allType6, error_queue))
        p7 = Process(target=self.fund_info_pzdata, args=(allCode7, allType7, error_queue))
        p8 = Process(target=self.fund_info_pzdata, args=(allCode8, allType8, error_queue))
        print('等待所有子进程完成。')
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()

        error_flag = error_queue.get()
        if error_flag is not None:
            print("error_flag,", error_flag)

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        p8.join()
        end = time.time()
        print("总共用时{}秒".format((end - start)))

    def run_reptile_pinzhongdata_single(self):
        error_queue = Queue()
        allCodeType = ReptileFund().get_all_code_type()

        allCode1 = allCodeType.loc[1002:1005, 'code'].values.tolist()
        allType1 = allCodeType.loc[1002:1005, 'type'].values.tolist()
        self.fund_info_pzdata(allCode1, allType1, error_queue)

    def run_reptile_pinzhongdata_test(self):
        self.single_code('000092', '混合型')
