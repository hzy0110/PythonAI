from dao.conn import Conn


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

    def test(self):
        with self.engineFunds.connect() as connFunds, connFunds.begin():
            sqlStatus = connFunds.execute(
                'insert into fund_info(fund_code,fund_name,fund_type) value(:1 ,:2 ,:3 ) on duplicate key update fund_code=:1, fund_name=:2, fund_type=:3',
                (3, 'aaa', 999)
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

    def save_manager_info(self, manager_info_pd):
        # insert,update 模式下，需要插入where 信息
        # print(manager_info_pd)
        # print(manager_info_pd.shape)
        manager_info_pd.fillna(0, inplace=True)
        manager_info_pd.replace(to_replace=r'^\s*$', value=0, regex=True, inplace=True)
        sql_str = self.get_insert_update_sql(manager_info_pd, 'manager_info')
        # print(sql_str)
        exec_info = manager_info_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
        return rowcount, is_insert

    def save_manager_current(self, manager_current_pd):
        # insert,update 模式下，需要插入where 信息
        manager_current_pd.fillna(0, inplace=True)
        manager_current_pd.replace(to_replace=r'^\s*$', value=0, regex=True, inplace=True)
        sql_str = self.get_insert_update_sql(manager_current_pd, 'manager_current')
        # print(sql_str)
        exec_info = manager_current_pd.apply(
            lambda x: self.execute_by_lambda_double_value_list(self.engineFunds.connect(), sql_str, x.tolist()), axis=1)
        rowcount = exec_info[0].rowcount
        is_insert = exec_info[0].is_insert
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
