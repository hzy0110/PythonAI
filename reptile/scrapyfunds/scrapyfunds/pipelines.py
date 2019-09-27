# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import os
import pandas as pd
import numpy as np


class ScrapyfundsPipeline(object):
    def process_item(self, item, spider):
        if spider.name == 'FundsGrandTotalReportPer':
            if not item['fund_grant_total_pd'].empty:
                fund_grant_total_pd = item['fund_grant_total_pd']
                code = item['code']
                # if not os.path.exists('./data/fund/fund_grant_total_all_pd.csv'):
                fund_grant_total_pd.to_csv('./data/fund/gt/' + code + '.csv', encoding='utf_8_sig', index=False)
                # else:
                #     fund_grant_total_history_pd = pd.read_csv('./data/fund/fund_grant_total_all_pd.csv', low_memory=False)
                #     fund_grant_total_history_pd = pd.merge(fund_grant_total_history_pd, fund_grant_total_pd, how='left',
                #                                            on='日期')
                #     fund_grant_total_history_pd.to_csv('./data/fund/fund_grant_total_all_pd.csv', encoding='utf_8_sig', index=False)

        if spider.name == 'FundsJAJXRating':
            if not item['fund_jajx_rating_pd'].empty:
                fund_jajx_rating_pd = item['fund_jajx_rating_pd']
                if not os.path.exists('./data/fund/fund_jajx_rating_pd.csv'):
                    fund_jajx_rating_pd.to_csv('./data/fund/fund_jajx_rating_pd.csv', encoding='utf_8_sig', index=False)
                else:
                    fund_jajx_rating_pd.to_csv('./data/fund/fund_jajx_rating_pd.csv', mode='a', header=False,
                                               encoding='utf_8_sig', index=False)

        if spider.name == 'FundsZSZQRating':
            if not item['fund_zszq_rating_pd'].empty:
                fund_zszq_rating_pd = item['fund_zszq_rating_pd']
                if not os.path.exists('./data/fund/fund_zszq_rating_pd.csv'):
                    fund_zszq_rating_pd.to_csv('./data/fund/fund_zszq_rating_pd.csv', encoding='utf_8_sig', index=False)
                else:
                    fund_zszq_rating_pd.to_csv('./data/fund/fund_zszq_rating_pd.csv', mode='a', header=False,
                                               encoding='utf_8_sig', index=False)

        if spider.name == 'FundsSHZQRating':
            if not item['fund_shzq_rating_pd'].empty:
                fund_rating_pd = item['fund_shzq_rating_pd']
                if not os.path.exists('./data/fund/fund_shzq_rating_pd.csv'):
                    fund_rating_pd.to_csv('./data/fund/fund_shzq_rating_pd.csv', encoding='utf_8_sig', index=False)
                else:
                    fund_rating_pd.to_csv('./data/fund/fund_shzq_rating_pd.csv', mode='a', header=False,
                                          encoding='utf_8_sig', index=False)

        if spider.name == 'FundsRating':
            if not item['fund_rating_pd'].empty:
                fund_rating_pd = item['fund_rating_pd']
                if not os.path.exists('./data/fund/fund_rating_pd.csv'):
                    fund_rating_pd.to_csv('./data/fund/fund_rating_pd.csv', encoding='utf_8_sig', index=False)
                else:
                    fund_rating_pd.to_csv('./data/fund/fund_rating_pd.csv', mode='a', header=False,
                                          encoding='utf_8_sig', index=False)

        if spider.name == 'FundsManageFeatureData':
            if not item['feature_data_pd'].empty:
                feature_data_pd = item['feature_data_pd']
                if not os.path.exists('./data/fund/feature_data_pd.csv'):
                    feature_data_pd.to_csv('./data/fund/feature_data_pd.csv', encoding='utf_8_sig', index=False)
                else:
                    feature_data_pd.to_csv('./data/fund/feature_data_pd.csv', mode='a', header=False,
                                           encoding='utf_8_sig', index=False)

        if spider.name == 'FundsManageSimilar':
            if item['manager_pd']:
                manager_pd = item['manager_pd']
                manager_id = item['manager_id']
                for m_pd, m_id in zip(manager_pd, manager_id):
                    # print("m_id", type(m_id))
                    manager_id_list = np.loadtxt("./data/temp/manager_id_np.csv", dtype=np.int, delimiter=',').tolist()
                    # if len(np.atleast_1d(manager_id_list)) > 1:
                    if int(m_id) not in manager_id_list:
                        if not os.path.exists('./data/manager/manager_history_similar_pd.csv'):
                            m_pd.to_csv('./data/manager/manager_history_similar_pd.csv', encoding='utf_8_sig',
                                        index=False)
                        else:
                            m_pd.to_csv('./data/manager/manager_history_similar_pd.csv', mode='a', header=False,
                                        encoding='utf_8_sig', index=False)
                        manager_id_list.append(m_id)
                        # print('manager_id_list', manager_id_list)
                        np.savetxt('./data/temp/manager_id_np.csv', np.array(manager_id_list), fmt="%s", delimiter=',')

            if not item['fund_history_manager_pd'].empty:
                fund_history_manager_pd = item['fund_history_manager_pd']
                if not os.path.exists('./data/fund/fund_history_manager_pd.csv'):
                    fund_history_manager_pd.to_csv('./data/fund/fund_history_manager_pd.csv', encoding='utf_8_sig',
                                                   index=False)
                else:
                    fund_history_manager_pd.to_csv('./data/fund/fund_history_manager_pd.csv', mode='a', header=False,
                                                   encoding='utf_8_sig', index=False)
            return item
        if spider.name == 'fundsManageTypeList':
            if item['manager_pd']:
                manager_history_pd = item['manager_pd'][0]
                manager_current_pd = item['manager_pd'][1]
                # manager_history_pd
                if not os.path.exists('./data/manager/manager_history_pd.csv'):
                    manager_history_pd.to_csv('./data/manager/manager_history_pd.csv', encoding='utf_8_sig',
                                              index=False)
                else:
                    manager_history_pd.to_csv('./data/manager/manager_history_pd.csv', mode='a', header=False,
                                              encoding='utf_8_sig', index=False)
                # manager_history_pd
                if not os.path.exists('./data/manager/manager_current_pd.csv'):
                    manager_current_pd.to_csv('./data/manager/manager_current_pd.csv', encoding='utf_8_sig',
                                              index=False)
                else:
                    manager_current_pd.to_csv('./data/manager/manager_current_pd.csv', mode='a', header=False,
                                              encoding='utf_8_sig', index=False)

            return item
