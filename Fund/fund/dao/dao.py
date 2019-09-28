from fund.conn import Conn


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

    def update_threshold_value(self, tvID, value):
        with self.engineDataV.connect() as connDataV, connDataV.begin():
            sqlStatus = connDataV.execute(
                'update threshold set ThresholdValue = %s where ID = %s',
                [value, tvID]
            )
        return sqlStatus
