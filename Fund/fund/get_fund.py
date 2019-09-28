from fund.tools.reptile import Reptile
from fund.tools.tool import Tool
from fund.dao.dao import Dao
import time
import sys
from multiprocessing import Process


class Fund:
    def __init__(self):
        pass

    def all_code_m(self, allCode, allType):
        for i in range(len(allCode)):
            code, ishb, basic_info_pd, funds = Reptile().get_pingzhongdata(allCode[i], allType[i])
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
                sys.stdout.flush()
        print("end")

    def run_reptile_pinzhongdata_multiple(self):
        allCodeType = Reptile().get_all_code_type()

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
        p1 = Process(target=self.all_code_m, args=(allCode1, allType1))
        p2 = Process(target=self.all_code_m, args=(allCode2, allType2))
        p3 = Process(target=self.all_code_m, args=(allCode3, allType3))
        p4 = Process(target=self.all_code_m, args=(allCode4, allType4))
        p5 = Process(target=self.all_code_m, args=(allCode5, allType5))
        p6 = Process(target=self.all_code_m, args=(allCode6, allType6))
        p7 = Process(target=self.all_code_m, args=(allCode7, allType7))
        p8 = Process(target=self.all_code_m, args=(allCode8, allType8))
        print('等待所有子进程完成。')
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()
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
