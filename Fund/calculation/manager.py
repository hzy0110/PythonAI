import pandas as pd
from dao.dao import Dao

class CalManager:
    def __init__(self):
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

    # 以下方法无用
    # 计算经理擅长的类型
    def get_manager_good_at_type(self):
        pass
        manager_good_at_type_pd = Dao().get_manager_good_at_type()
        # manager_good_at_type_pd.groupby(
        #                 manager_good_at_type_pd).count().idxmax()
        # print(manager_good_at_type_pd)
        # 计算经理擅长的类型，并保存
        # allManager_id_s = allManager_pd['manageID']
        # m_id = []
        # m_gat = []
        # for manage_id in allManager_id_s:
        #     fund_type_history_s = manager_history_pd.loc[manager_history_pd['经理ID'] ==
        #                                                  manage_id]['基金类型']
        #     fund_type_current_s = manager_current_pd.loc[manager_current_pd['经理ID'] ==
        #                                                  manage_id]['基金类型']
        #     # 合并类型
        #     fund_type_all_s = fund_type_history_s.append(fund_type_current_s)
            #     print(manage_id,fund_type_all_s.shape)
            # 获取类型最多的，作为擅长类型
        #     if len(fund_type_all_s) > 0:
        #         manager_goo_at_type = fund_type_all_s.groupby(
        #             fund_type_all_s).count().idxmax()
        #         m_id.append(manage_id)
        #         m_gat.append(manager_goo_at_type)
        #
        # manager_goo_at_type_pd = pd.DataFrame(columns=['manageID', 'good_at_type'])
        # manager_goo_at_type_pd['manageID'] = m_id
        # manager_goo_at_type_pd['good_at_type'] = m_gat
        # allManager_good_at_type_pd = pd.merge(
        #     allManager_pd, manager_goo_at_type_pd, how='left', on='manageID')
        # allManager_good_at_type_pd.to_csv(
        #     './data/manager/allManager_pd.csv',
        #     encoding='utf_8_sig',
        #     index=False)
        # allManager_pd = pd.read_csv('./data/manager/allManager_pd.csv', low_memory=False)




    # 计算经理数据
    def get_manager_aaa(self, manager_history_all_pd):
        manager_history_all_pd['同类排名比'] = manager_history_all_pd.apply(
            lambda x: self.similarPer(x['同类排名']), axis=1)

        manager_history_all_pd['任职同类差'] = manager_history_all_pd.apply(
            lambda x: x['任职回报'] - x['同类平均'], axis=1)

        manager_history_all_pd['同类差比和'] = manager_history_all_pd.apply(
            lambda x: x['同类排名比'] + x['任职同类差'], axis=1)

        manager_history_all_pd['同类差比乘'] = manager_history_all_pd.apply(
            lambda x: x['同类排名比'] * x['任职同类差'], axis=1)


    def get_near_year_aaa(self, allManager_pd, manager_history_all_pd):
        mID_s = allManager_pd['manageID']
        for y in range(2, 6, 1):
            print("执行到", y, "年")
            m_history_1_list = []
            m_history_2_list = []
            m_history_3_list = []
            m_history_4_list = []
            for mID in mID_s:
                m_history = manager_history_all_pd.loc[manager_history_all_pd['经理ID'] ==
                                                       mID]

                m_history_1 = m_history.loc[m_history['任职天数'] > y * 365]['同类排名比'].sum()
                m_history_1_list.append(m_history_1)
                m_history_2 = m_history.loc[m_history['任职天数'] > y * 365]['任职同类差'].sum()
                m_history_2_list.append(m_history_2)
                m_history_3 = m_history.loc[m_history['任职天数'] > y * 365]['同类差比和'].sum()
                m_history_3_list.append(m_history_3)
                m_history_4 = m_history.loc[m_history['任职天数'] > y * 365]['同类差比乘'].sum()
                m_history_4_list.append(m_history_4)

            allManager_pd.insert(allManager_pd.shape[1], '近' + str(y) + '年内同类排名比', m_history_1_list)
            allManager_pd.insert(allManager_pd.shape[1], '近' + str(y) + '年内任职同类差', m_history_2_list)
            allManager_pd.insert(allManager_pd.shape[1], '近' + str(y) + '年内同类差比和', m_history_3_list)
            allManager_pd.insert(allManager_pd.shape[1], '近' + str(y) + '年内同类差比乘', m_history_4_list)

    def get_near_year_2(self, allManager_pd):
        # 总表计算各年的总和
        # allManager_pd = pd.read_csv('./data/manager/allManager_1_pd.csv', low_memory=False)
        for y in range(2, 6, 1):
            allManager_pd['近' + str(y) + '年内总分'] = allManager_pd.apply(
                lambda r:
                r['近' + str(y) + '年内同类排名比'] +
                r['近' + str(y) + '年内任职同类差'] +
                r['近' + str(y) + '年内同类差比和'] +
                r['近' + str(y) + '年内同类差比乘'], axis=1)
        allManager_pd.to_csv('./data/manager/allManager_2_pd.csv', index=False, encoding='utf_8_sig')
