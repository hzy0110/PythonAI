# from fund.tools.reptile import Reptile
# from fund.tools.tool import Tool
# from fund.dao.dao import Dao
from run.fund import Fund
from run.manager import Manager


# 运行部分
# 爬虫
# 需要写一个覆盖基金类型的方法，手动测试类型，会导致数据不准确
# Fund().run_reptile_pinzhongdata_multiple()
# Fund().run_reptile_pinzhongdata_test()
# Fund().fund_history_feature()
# Fund().fund_info_feature()
Fund().fund_base_info()
# Fund().run_reptile_pinzhongdata_single()
# Fund().fund_info_shzq_star()
# Fund().fund_info_zszq_star()
# Fund().fund_info_jajx_star()
# Fund().fund_history_manager()

# Manager().run_menager()
# Manager().manager_current_manager_page()
# Manager().manager_history_by_fund_manager_page()

# 处理数据
# Manager().manager_get_good_at_type()
# Fund().fund_info_parameter()

# if __name__ == "__main__":
#     main()
