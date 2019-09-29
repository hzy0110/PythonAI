# from fund.tools.reptile import Reptile
# from fund.tools.tool import Tool
# from fund.dao.dao import Dao
from run.fund import Fund
from run.manager import Manager
# def main():

# allCodeType = Reptile().get_all_code_type()
# stsrt = 0
# f = open('./data/endindex', 'r', encoding='utf-8')
# stsrt = int(f.read())
#
# allCode = allCodeType.loc[stsrt:, 'code'].values.tolist()
# allType = allCodeType.loc[stsrt:, 'type'].values.tolist()
# for i in range(len(allCode)):
#     # print('i=', i)
#     start = time.time()
#     # ishb = False
#     code, ishb, basic_info_pd, fund = Reptile().get_pingzhongdata(allCode[i], allType[i])
#     end = time.time() - start
#     print(code, '用时1', end)
#     start = time.time()
#     if ishb is not None:
#         print("正在执行：", code)
#         if ishb:
#             fund_wide_pd, rateInSimilar_pd, grandTotal_pd, worth_pd, current_fund_manager_pd = Tool().get_hb_pingzhongdata_2_df(
#                  basic_info_pd, fund)
#         else:
#             fund_wide_pd, rateInSimilar_pd, grandTotal_pd, worth_pd, current_fund_manager_pd = Tool().get_pingzhongdata_2_df(
#                  basic_info_pd, fund)
#         end = time.time() - start
#         print(code, '用时2', end)
#         start = time.time()
#         Dao().save_fund_info(fund_wide_pd, 'fund_code', int(code))
#         end = time.time() - start
#         print(code, '用时3', end)
#         start = time.time()
#         if not rateInSimilar_pd.empty:
#             Dao().save_fund_history(rateInSimilar_pd, 'fund_code', int(code))
#             pass
#         if not grandTotal_pd.empty:
#             Dao().save_fund_history(grandTotal_pd, 'fund_code', int(code))
#             pass
#         if not worth_pd.empty:
#             Dao().save_fund_history(worth_pd, 'fund_code', int(code))
#             pass
#
#         f = open('./data/endindex', 'w+', encoding='utf-8')
#         f.write(str(stsrt + i))
#         end = time.time() - start
#         print(code, '用时4', end)
#         sys.stdout.flush()
#     # Dao().test()
#
# Manager().run_menager()
# Fund().run_reptile_pinzhongdata_multiple()
# 需要写一个覆盖基金类型的方法，手动测试类型，会导致数据不准确
# Fund().run_reptile_pinzhongdata_test()
Fund().fund_info_feature()
# Fund().run_reptile_pinzhongdata_single()
# if __name__ == "__main__":
#     main()
