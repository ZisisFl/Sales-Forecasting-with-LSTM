import pandas
import numpy
from datetime import date

DATA_TYPE = '.csv'
INPUT_PATH = 'data/raw/'
OUTPUT_PATH = 'data/processed/'

dataframe = pandas.read_csv(INPUT_PATH + 'sales_train_v2' + DATA_TYPE, parse_dates=['date'], infer_datetime_format=True,
                            dayfirst=True)

dataframe = dataframe[dataframe['item_cnt_day'] > 0]
print('List of product frequency\n', dataframe['item_id'].value_counts())
# drop column price
dataframe = dataframe.drop(['item_price'], axis=1)

# pick a product
PRODUCT_ID = 20949
dataframe = dataframe[dataframe['item_id'] == PRODUCT_ID]
print(dataframe)

number_of_shops = dataframe['shop_id'].nunique()
u_item_list = dataframe.item_id.unique().tolist()
date_range = pandas.date_range(start='1/1/2013', end='31/10/2015')

d0 = date(2013, 1, 1)
d1 = date(2015, 10, 31)
delta = (d1 - d0).days
# print(delta)

week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
significant_days = ['2013-11-29', '2014-11-28', '2013-12-23', '2013-12-24', '2014-12-23', '2014-12-24']

# df_days: is a one hot representation of the week
df_days = pandas.DataFrame(numpy.random.randint(low=0, high=1, size=(7, delta+1)), index=week)

for column in df_days:  # 1/1/2013 is Tuesday so every day with column % 7 == 0 will be Tuesday
    if column % 7 == 0:
        days_value = week[1]
        df_days.at[days_value, column] = 1
    elif column % 7 == 1:
        days_value = week[2]
        df_days.at[days_value, column] = 1
    elif column % 7 == 2:
        days_value = week[3]
        df_days.at[days_value, column] = 1
    elif column % 7 == 3:
        days_value = week[4]
        df_days.at[days_value, column] = 1
    elif column % 7 == 4:
        days_value = week[5]
        df_days.at[days_value, column] = 1
    elif column % 7 == 5:
        days_value = week[6]
        df_days.at[days_value, column] = 1
    elif column % 7 == 6:
        days_value = week[0]
        df_days.at[days_value, column] = 1

df_days.columns = date_range

# df_s_days: is a dataframe in which days of high significance like black friday have value 1
df_s_days = pandas.DataFrame(numpy.random.randint(low=0, high=1, size=(1, delta+1)), index=['S_Day'],
                             columns=date_range)

for i in range(len(significant_days)):
    df_s_days.at['S_Day', significant_days[i]] = 1


sales = pandas.DataFrame()
for i in range(60):
    df = dataframe[dataframe['shop_id'] == i]

    # df_items: dataframe that contains number of sales for every item everyday
    df_items = pandas.DataFrame(index=u_item_list, columns=date_range)
    df_items = (df.pivot('item_id', 'date', 'item_cnt_day').reindex(index=df_items.index, columns=df_items.columns).
                fillna(0))
    # print(df_items)

    # result: is a df that contains values, dates, one hot rep weekdays and significant days for every shop
    result = pandas.concat([df_items, df_days, df_s_days])
    sales = pandas.concat([sales, result], axis=1)

sales = sales.T
print(sales)

sales.to_csv(OUTPUT_PATH + 'ALL_SHOPS_PRODUCT_'+str(PRODUCT_ID) + DATA_TYPE, encoding='utf-8', index=True, header=True)
