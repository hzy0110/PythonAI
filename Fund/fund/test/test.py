import pandas as pd
#
# test_pd = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a1', 'a2', 'a3'])
# columns_list = test_pd.columns.tolist()
# print(columns_list)
# # update fund_info set fund_name ='aaaa' , fund_type = 2 where fund_code = 1
# table_name ='fund_name'
# sql_str =  'update '+table_name+' set '
# 'ruleexelasttime = :1 where labelfatherid = :2'
# where_str = 'a3'
# where_value = '5'
# for i in range(len(columns_list)):
#
#     sql_str += columns_list[i] + '=%s '
#     if i != len(columns_list) - 1:
#         sql_str += ','
#
# sql_str = sql_str + ' where ' + where_str + '=' +where_value
# print(sql_str)
table_name = 'fund_info'
where_columns = 'fund_code'
where_value = 3

df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['fund_code', 'fund_name', 'fund_type'])
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
sql_str = 'insert into ' + table_name + '('+insert_columns_str+') value(' + insert_value_str + ')'
sql_str = sql_str + ' on duplicate key update '
sql_str = sql_str + update_str
# sql_str = sql_str + ' , ' + where_columns + '=' + str(where_value)
print(sql_str)
# print(insert_columns_str)
list1 = [1,2,3]

list1  = list1+ list1
print(list1)