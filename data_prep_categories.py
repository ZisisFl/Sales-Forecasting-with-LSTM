import pandas
import numpy as np
from datetime import date

DATA_TYPE = '.csv'
INPUT_PATH = 'data/raw/'
OUTPUT_PATH = 'data/processed/'

dataframe = pandas.read_csv(INPUT_PATH + 'sales_train_v2' + DATA_TYPE, parse_dates=['date'], infer_datetime_format=True,
                            dayfirst=True)

# drop column price
dataframe = dataframe.drop(['item_price'], axis=1)

# create a dataframe with categorie
item_cat = pandas.read_csv(INPUT_PATH + 'items' + DATA_TYPE)

# merge categories with the items
dataframe = pandas.merge(dataframe, item_cat, on='item_id')
dataframe = dataframe.drop(['item_name', 'date_block_num', 'item_id'], axis=1)
dataframe = dataframe[dataframe['item_cnt_day'] > 0]

# create list with unique category ids and their count
unique_category_list = dataframe.item_category_id.unique().tolist()
number_of_unique_categories = item_cat['item_category_id'].nunique()
# print(unique_category_list)
# print(number_of_unique_categories)

# sum number of sales from items in the same category
dataframe = dataframe.groupby(['date', 'shop_id', 'item_category_id'], as_index=False)['item_cnt_day'].sum()
# print(dataframe)

# generate the number of days
d0 = date(2013, 1, 1)
d1 = date(2015, 10, 31)
delta = (d1 - d0).days

date_range = pandas.date_range(start='1/1/2013', end='31/10/2015')
week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
significant_days = ['2013-11-29', '2014-11-28', '2013-12-23', '2013-12-24', '2014-12-23', '2014-12-24']


# df_days: is a one hot representation of the week
df_days = pandas.DataFrame(np.random.randint(low=0, high=1, size=(7, delta+1)), index=week)

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
df_s_days = pandas.DataFrame(np.random.randint(low=0, high=1, size=(1, delta+1)), index=['S_Day'],
                             columns=date_range)


for i in range(len(significant_days)):
    df_s_days.at['S_Day', significant_days[i]] = 1

# create a dataframe with the category sales from all shops
category_sales = pandas.DataFrame()
for i in range(60):
    df = dataframe[dataframe['shop_id'] == i]

    # df_items: dataframe that contains number of sales for every item everyday
    df_categories = pandas.DataFrame(index=unique_category_list, columns=date_range)
    df_categories = (df.pivot('item_category_id', 'date', 'item_cnt_day')
                     .reindex(index=df_categories.index, columns=df_categories.columns).fillna(0))
    result = pandas.concat([df_categories, df_days, df_s_days])
    category_sales = pandas.concat([category_sales, result], axis=1)

category_sales = category_sales.T
# print(category_sales)
category_sales.to_csv(OUTPUT_PATH + 'CATEGORIES_ALL_SHOPS' + DATA_TYPE, encoding='utf-8', index=True, header=True)