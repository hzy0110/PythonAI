from fund.fund_info import FundInfo
from fund.tools.reptile import Reptile
from fund.tools.tool import Tool
from fund.dao.dao import Dao
import sys
import js2py
import execjs
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():
    start = time.time()
    allCodeType = Reptile().get_all_code_type()
    end = time.time() - start
    print('用时', end)
    allCode = allCodeType.loc[1:2, 'code'].values.tolist()
    allType = allCodeType.loc[1:2, 'type'].values.tolist()
    for i in range(len(allCode)):
        code, basic_info_pd, funds = Reptile().get_pingzhongdata(allCode[i], allType[i])
        if code is not None:
            fund_wide_pd, rateInSimilar_pd, grandTotal_pd, worth_pd, current_fund_manager_pd = Tool().get_pingzhongdata_2_df(
                code, basic_info_pd, funds)
            Dao().save_fund_info(fund_wide_pd, 'fund_code', int(code))
            Dao().save_fund_history(worth_pd, 'fund_code', int(code))
            Dao().save_fund_history(grandTotal_pd, 'fund_code', int(code))
            Dao().save_fund_history(rateInSimilar_pd, 'fund_code', int(code))

    # Dao().test()
#

if __name__ == "__main__":
    main()
