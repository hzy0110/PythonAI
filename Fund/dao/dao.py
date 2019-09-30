from dao.conn import Conn
import pandas as pd

class Dao:
    def __init__(self):
        self.engineFunds = Conn().getEngineFunds()

    def insert_fund_code(self, fond_code):
        with self.engineFunds.connect() as connFunds, connFunds.begin():
            sqlStatus = connFunds.execute(
                'INSERT INTO fund_info(fund_code) VALUES (%s)',
                [fond_code]
            )
            return sqlStatus

    def save_fund_info(self, fund_info_pd, where_columns, where_value):
        # 替换空值和空字符串
        fund_info_pd.fillna(0, inplace=True)
        fund_info_pd.replace(to_replace=r'^\s*$', value=0, regex=True, inplace=True)
        sql_str = self.get_update_sql(fund_info_pd, 'fund_info', where_columns, where_value)
        exec_info = fund_info_pd.apply(
            lambda x: self.execute_by_lambda(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def save_fund_info(self, fund_info_pd):
        # 替换空值和空字符串
        fund_info_pd.fillna(0, inplace=True)
        fund_info_pd.replace(to_replace=r'^\s*$', value=0, regex=True, inplace=True)
        sql_str = self.get_insert_update_sql(fund_info_pd, 'fund_info')
        exec_info = fund_info_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def save_fund_history(self, fund_history_pd, where_columns, where_value):
        # insert,update 模式下，需要插入where 信息
        fund_history_pd.insert(0, where_columns, where_value)
        fund_history_pd.fillna(0, inplace=True)
        fund_history_pd.replace(to_replace=r'^\s*$', value=0, regex=True, inplace=True)
        sql_str = self.get_insert_update_sql(fund_history_pd, 'fund_history')
        # print(sql_str)
        exec_info = fund_history_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def save_fund_history(self, fund_history_pd):
        fund_history_pd.fillna(0, inplace=True)
        fund_history_pd.replace(to_replace=r'^\s*$', value=0, regex=True, inplace=True)
        sql_str = self.get_insert_update_sql(fund_history_pd, 'fund_history')
        # print(sql_str)
        exec_info = fund_history_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def save_fund_info_by_db(self, fund_info_pd):
        sql_str = self.get_insert_update_sql(fund_info_pd, 'fund_info')
        exec_info = fund_info_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def save_fund_history_by_db(self, fund_history_pd):
        sql_str = self.get_insert_update_sql(fund_history_pd, 'fund_history')
        exec_info = fund_history_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def save_fund_history_manager(self, fund_history_manager_pd):
        fund_history_manager_pd.fillna(0, inplace=True)
        fund_history_manager_pd.replace(to_replace=r'^\s*$', value=0, regex=True, inplace=True)
        sql_str = self.get_insert_update_sql(fund_history_manager_pd, 'fund_history_manager')
        # print(sql_str)
        exec_info = fund_history_manager_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def get_fund_info_by_fund_code(self, fund_code):
        with self.engineFunds.connect() as connFunds, connFunds.begin():
            return pd.read_sql_query(
                """
                select * from fund_info as f 
                where f.fund_code = %s
                """,
                params=([fund_code]),  con=connFunds)

    def get_fund_history_max_trade_timestamp_by_fund_code(self, fund_code):
        with self.engineFunds.connect() as connFunds, connFunds.begin():
            return pd.read_sql_query(
                """
                select max(f.trade_timestamp) as max_trade_timestamp from fund_history as f 
                where f.fund_code = %s
                """,
                params=([fund_code]),  con=connFunds)

    def get_fund_history_by_fund_code_trade_timestamp(self, fund_code, trade_timestamp):
        with self.engineFunds.connect() as connFunds, connFunds.begin():
            return pd.read_sql_query(
                """
                SELECT f.* from fund_history as f where f.fund_code=%s and f.trade_timestamp=%s
                """,
                params=([fund_code, trade_timestamp]), con=connFunds)

    def get_fund_history_by_trade_timestamp_interval(self, fund_code, start_trade_timestamp, end_trade_timestamp):
        with self.engineFunds.connect() as connFunds, connFunds.begin():
            return pd.read_sql_query(
                """
                SELECT * from fund_history as f where f.fund_code = %s and
                 ( f.trade_timestamp>=%s and f.trade_timestamp<=%s)
                """,
                params=([fund_code, start_trade_timestamp, end_trade_timestamp]), con=connFunds)

    def get_dividend_count_by_trade_timestamp_interval(self, fund_code, start_trade_timestamp, end_trade_timestamp):
        with self.engineFunds.connect() as connFunds, connFunds.begin():
            return pd.read_sql_query(
                """
                SELECT count(0) as dividend_count from fund_history as f where f.fund_code = %s and
                 ( f.trade_timestamp>=%s and f.trade_timestamp<=%s)
                 and unit_money > 0
                """,
                params=([fund_code, start_trade_timestamp, end_trade_timestamp]), con=connFunds)

    def get_dividend_amount_by_trade_timestamp_interval(self, fund_code, start_trade_timestamp, end_trade_timestamp):
        with self.engineFunds.connect() as connFunds, connFunds.begin():
            return pd.read_sql_query(
                """
                SELECT sum(unit_money) as dividend_amount from fund_history as f where f.fund_code = %s and
                 ( f.trade_timestamp>=%s and f.trade_timestamp<=%s)
                 and unit_money > 0
                """,
                params=([fund_code, start_trade_timestamp, end_trade_timestamp]), con=connFunds)

    def save_manager_info(self, manager_info_pd):
        manager_info_pd.fillna(0, inplace=True)
        manager_info_pd.replace(to_replace=r'^\s*$', value=0, regex=True, inplace=True)
        sql_str = self.get_insert_update_sql(manager_info_pd, 'manager_info')
        exec_info = manager_info_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def save_manager_current(self, manager_current_pd):
        # 清空表，全量灌入
        table_name = 'manager_current'
        self.truncate(self.engineFunds.connect(), table_name)

        manager_current_pd.fillna(0, inplace=True)
        manager_current_pd.replace(to_replace=r'^\s*$', value=0, regex=True, inplace=True)
        sql_str = self.get_insert_update_sql(manager_current_pd, table_name)
        # print(sql_str)
        exec_info = manager_current_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def save_manager_history(self, manager_history_pd):
        manager_history_pd.fillna(0, inplace=True)
        manager_history_pd.replace(to_replace=r'^\s*$', value=0, regex=True, inplace=True)
        sql_str = self.get_insert_update_sql(manager_history_pd, 'manager_history')
        # print(sql_str)
        exec_info = manager_history_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def get_manager_good_at_type(self):
        with self.engineFunds.connect() as connFunds, connFunds.begin():
            sqlStatus = connFunds.execute(
                """
                update manager_info as m,(
                    SELECT e.manager_code,e.fund_type as good_at_type FROM
                            (SELECT manager_code, max(s) s FROM 
                                (SELECT b.* FROM
                                    (SELECT manager_code, max(c) as  c  , max(s)  as s FROM
                                        (SELECT manager_code,fund_type,count(0) as c,sum(job_report)  as s
                                            from manager_history  GROUP BY manager_code,fund_type
                                        ) as c 
                                    GROUP BY manager_code ) a
                                INNER JOIN 
                                    (SELECT manager_code,fund_type,count(0) as c,sum(job_report)  as s
                                        from manager_history   GROUP BY manager_code,fund_type
                                    ) as b
                                ON a.manager_code = b.manager_code
                                AND a.c = b.c ) as g
                            GROUP BY manager_code ) f
                        INNER JOIN 
                            (SELECT b.* FROM
                                (SELECT manager_code, max(c) as  c  , max(s)  as s FROM
                                    (SELECT manager_code,fund_type,count(0) as c,sum(job_report)  as s
                                        from manager_history   GROUP BY manager_code,fund_type
                                    ) as c 
                                GROUP BY manager_code ) a
                            INNER JOIN 
                                (SELECT manager_code,fund_type,count(0) as c,sum(job_report)  as s
                                from manager_history   GROUP BY manager_code,fund_type
                                ) as b
                            ON a.manager_code = b.manager_code
                            AND a.c = b.c ) as e 
                        ON f.manager_code = e.manager_code
                        AND f.s = e.s
                    ) as h
                set m.good_at_type=h.good_at_type
                where m.manager_code=h.manager_code;
                """)
            print(type(sqlStatus), sqlStatus)
            rowcount = sqlStatus.rowcount
            is_insert = sqlStatus.is_insert
            return rowcount, is_insert

    def get_update_sql(self, df, table_name, where_columns, where_value):
        columns_list = df.columns.tolist()
        sql_str = 'update ' + table_name + ' set '
        for i in range(len(columns_list)):
            sql_str += columns_list[i] + '=%s '
            if i != len(columns_list) - 1:
                sql_str += ','

        sql_str = sql_str + ' where ' + where_columns + '=' + str(where_value)
        return sql_str

    def get_insert_update_sql(self, df, table_name):
        columns_list = df.columns.tolist()
        insert_columns_str = ''
        insert_value_str = ''
        update_str = ''

        for i in range(len(columns_list)):
            insert_columns_str += columns_list[i]
            insert_value_str += '%s '
            if i != len(columns_list) - 1:
                insert_columns_str += ','
                insert_value_str += ','

        for i in range(len(columns_list)):
            update_str += columns_list[i] + '=%s'
            if i != len(columns_list) - 1:
                update_str += ', '
        sql_str = 'insert into ' + table_name + '(' + insert_columns_str + ') value(' + insert_value_str + ')'
        sql_str = sql_str + ' on duplicate key update '
        sql_str = sql_str + update_str
        # sql_str = sql_str + ' , ' + where_columns + '=' + str(where_value)
        return sql_str

    def truncate(self, engine, table_name):
        with engine, engine.begin():
            sqlStatus = engine.execute(
                "truncate table %s" %table_name
            )
            return sqlStatus


    def execute_by_lambda(self, engine, sql_str, value_list):
        with engine, engine.begin():
            sqlStatus = engine.execute(
                sql_str,
                value_list
            )
            return sqlStatus

    def execute_by_lambda_double_value_list(self, engine, sql_str, value_list):
        value_list = value_list+value_list
        # print(value_list)
        with engine, engine.begin():
            sqlStatus = engine.execute(
                sql_str,
                value_list
            )
            return sqlStatus
