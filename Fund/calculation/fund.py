import pandas as pd
from tools.tool import Tool
from reptile.fund import ReptileFund
from dao.dao import Dao
from tools.tool import Tool


class CalFund:
    def __init__(self):
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

    def cal_fund_parameter_2_max_trade(self):
        fund_code = 400001
        max_trade_timestamp = int(Dao().get_fund_history_max_trade_timestamp_by_fund_code(fund_code)['max_trade_timestamp'][0])
        fund_create_timestamp = int(Dao().fund_info(fund_code)['fund_create_timestamp'][0])
        date_list = ['30d', '90d', '180d', '1y', '2y', '3y', '4y', '5y']
        print('max_trade_timestamp', max_trade_timestamp)
        trade_day_timestamp_list = Tool().get_trade_day_list(max_trade_timestamp)
        print(trade_day_timestamp_list)
        max_trade_timestamp_fund_history_pd = Dao().get_fund_history_by_fund_code_trade_timestamp(
            fund_code, max_trade_timestamp)
        max_trade_timestamp_fund_history_pd.drop('created_time', inplace=True, axis=1)
        for timestamp, date in zip(trade_day_timestamp_list, date_list):
            print('timestamp', timestamp)
            if timestamp < fund_create_timestamp:
                timestamp = fund_create_timestamp
            self.save_fund_mean(max_trade_timestamp_fund_history_pd, fund_code, date, timestamp, max_trade_timestamp)
            self.save_fund_dividend(max_trade_timestamp_fund_history_pd, fund_code, date, timestamp, max_trade_timestamp)
            self.save_fund_report(max_trade_timestamp_fund_history_pd, fund_code, date, timestamp, max_trade_timestamp)
            self.save_fund_subtraction(max_trade_timestamp_fund_history_pd, fund_code, date, timestamp, max_trade_timestamp)
            # print(fund_history_pd.shape)

    def save_fund_mean(self, target_history_pd, fund_code, dateStr, start_trade_timestamp, end_trade_timestamp):
        interval_pd = Dao().get_fund_history_by_trade_timestamp_interval(fund_code, start_trade_timestamp,
                                                                         end_trade_timestamp)
        info_pd = pd.DataFrame([fund_code], columns=['fund_code'])
        info_pd['net_worth_report_' + dateStr + '_mean'] = round(interval_pd['equity_return'].mean(0), 10)
        info_pd['similar_ranking_' + dateStr + '_mean'] = round(interval_pd['similar_ranking'].mean(0), 10)
        info_pd['sc_' + dateStr + '_mean'] = round(interval_pd['sc'].mean(0), 10)
        info_pd['similar_ranking_per_' + dateStr + '_mean'] = round(
            interval_pd['similar_ranking_per'].mean(0), 10)
        info_pd['hs300_gt_earn_per_' + dateStr + '_mean'] = round(interval_pd['hs300_grand_total'].mean(0),
                                                                            10)
        info_pd['similar_mean_gt_earn_per_' + dateStr + '_mean'] = round(
            interval_pd['similar_mean_grand_total'].mean(0), 10)
        info_pd['self_gt_earn_per_' + dateStr + '_mean'] = round(interval_pd['self_grand_total'].mean(0), 10)
        info_pd['fund_code'] = fund_code

        self.save_fund_parameter(info_pd, end_trade_timestamp)

        # history_pd = info_pd.copy(deep=False)
        # history_pd['trade_timestamp'] = end_trade_timestamp
        #
        # rowcount, is_insert = Dao().save_fund_history_by_db(history_pd)
        # rowcount, is_insert = Dao().save_fund_info_by_db(info_pd)
        # return target_history_pd

    def save_fund_dividend(self, target_history_pd, fund_code, dateStr, start_trade_timestamp, end_trade_timestamp):
        dividend_amount = Dao().get_dividend_amount_by_trade_timestamp_interval(
            fund_code, start_trade_timestamp, end_trade_timestamp)['dividend_amount'][0]
        if dividend_amount is not None:
            dividend_amount_float = float(dividend_amount)
        else:
            dividend_amount_float = 0

        dividend_count = int(Dao().get_dividend_count_by_trade_timestamp_interval(
            fund_code, start_trade_timestamp, end_trade_timestamp)['dividend_count'][0])

        info_pd = pd.DataFrame([fund_code], columns=['fund_code'])
        info_pd['dividend_' + dateStr + '_amount'] = round(dividend_amount_float, 10)
        info_pd['dividend_' + dateStr + '_count'] = round(dividend_count, 10)
        self.save_fund_parameter(info_pd, end_trade_timestamp)

        # history_pd = info_pd.copy(deep=False)
        # history_pd['trade_timestamp'] = end_trade_timestamp
        #
        # rowcount, is_insert = Dao().save_fund_history_by_db(history_pd)
        # rowcount, is_insert = Dao().save_fund_info_by_db(info_pd)
        # print('save_fund_dividend_rowcount', rowcount)

    def save_fund_report(self, target_history_pd, fund_code, dateStr, start_trade_timestamp, end_trade_timestamp):
        start_pd = Dao().get_fund_history_by_fund_code_trade_timestamp(fund_code, start_trade_timestamp)
        report = target_history_pd.loc[0, 'net_worth'] - start_pd.loc[0, 'net_worth']/start_pd.loc[0, 'net_worth'] * 100

        info_pd = pd.DataFrame([fund_code], columns=['fund_code'])
        info_pd['near_report_' + dateStr + '_per'] = report

        self.save_fund_parameter(info_pd, end_trade_timestamp)

        # history_pd = info_pd.copy(deep=False)
        # history_pd['trade_timestamp'] = end_trade_timestamp

        # rowcount, is_insert = Dao().save_fund_history_by_db(history_pd)
        # rowcount, is_insert = Dao().save_fund_info_by_db(info_pd)
        # print('save_fund_dividend_rowcount', rowcount)

    def save_fund_subtraction(self, target_history_pd, fund_code, dateStr, start_trade_timestamp, end_trade_timestamp):
        start_pd = Dao().get_fund_history_by_fund_code_trade_timestamp(fund_code, start_trade_timestamp)
        similar_ranking_subtraction = start_pd.loc[0, 'similar_ranking'] - target_history_pd.loc[0, 'similar_ranking']
        sc_subtraction = start_pd.loc[0, 'sc'] - target_history_pd.loc[0, 'sc']

        info_pd = pd.DataFrame([fund_code], columns=['fund_code'])
        info_pd['similar_ranking_' + dateStr + '_change'] = similar_ranking_subtraction
        info_pd['sc_' + dateStr + '_change'] = sc_subtraction

        self.save_fund_parameter(info_pd, end_trade_timestamp)
        # history_pd = info_pd.copy(deep=False)
        # history_pd['trade_timestamp'] = end_trade_timestamp
        #
        # rowcount, is_insert = Dao().save_fund_history_by_db(history_pd)
        # rowcount, is_insert = Dao().save_fund_info_by_db(info_pd)
        # print('save_fund_dividend_rowcount', rowcount)

    def save_fund_parameter(self, info_pd, end_trade_timestamp):
        history_pd = info_pd.copy(deep=False)
        history_pd['trade_timestamp'] = end_trade_timestamp

        rowcount, is_insert = Dao().save_fund_history_by_db(history_pd)
        rowcount, is_insert = Dao().save_fund_info_by_db(info_pd)
        # print('save_fund_dividend_rowcount', rowcount)