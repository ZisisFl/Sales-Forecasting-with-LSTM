import pandas
import numpy as np
from matplotlib import pyplot as plt

DATA_TYPE = '.csv'
INPUT_PATH_P = 'data/processed/'
INPUT_PATH_R = 'data/raw/'

# General numbers from all data
sales_data = pandas.read_csv(INPUT_PATH_R + 'sales_train_v2' + DATA_TYPE)
print('General number from all data \n------------------------------')
print('number of shops:', sales_data['shop_id'].nunique())
print('number of items:', sales_data['item_id'].nunique())
print('number of months:', sales_data['date_block_num'].nunique())

category_data = pandas.read_csv(INPUT_PATH_R + 'item_categories' + DATA_TYPE)
print('number of categories:', category_data['item_category_name'].nunique())

date_range = pandas.date_range(start='1/1/2013', end='31/10/2015')
print('number of days', date_range.shape[0])


# create a plot of total sales for every Shop
sales_per_shop = sales_data.groupby(['shop_id'], as_index=False)['item_cnt_day'].sum()
# print(sales_per_shop)

plt.bar(sales_per_shop['shop_id'].tolist(), sales_per_shop['item_cnt_day'].tolist())
plt.xlabel('Shop id')
plt.ylabel('Number of total sales')
plt.title('Number of total sales for every Shop')
plt.show()
print('------------------------------')

# create plot of sales in every shop for every month
sales_per_shop_each_month = sales_data.groupby(['shop_id', 'date_block_num'], as_index=False)['item_cnt_day'].sum()
# print(sales_per_shop_each_month)

month_date_range = pandas.date_range(start='1/2013', end='10/2015', freq='MS').strftime("%Y-%b").tolist()
# print(month_date_range)

for i in range(60):
    df = sales_per_shop_each_month[sales_per_shop_each_month['shop_id'] == i]
    # print(df)
    month_range = []
    months = df['date_block_num'].tolist()
    for j in range(len(months)):
        month_range.append(month_date_range[j])
    plt.plot(month_range, df['item_cnt_day'].tolist(), label='shop'+str(i))


plt.xlabel('Time', rotation=0)
plt.ylabel('Number of total sales for each shop')
plt.title('Number of total sales for every Shop')
plt.xticks(rotation=45)
# plt.legend()
plt.grid()
plt.show()

# PS4 Stats
ps4_data = pandas.read_csv(INPUT_PATH_P + 'PS4_SET_ALL_SHOPS' + DATA_TYPE)
ps4_data = ps4_data.rename(index=str, columns={'Unnamed: 0': 'item_id'})
ps4_data = ps4_data.set_index('item_id')
# drop non item columns
ps4_data = ps4_data.drop(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'S_Day'], axis=1)
# print(ps4_data)

number_of_values = ps4_data.count().sum()
count_zero_values = ps4_data[ps4_data == 0].count().sum()
percetage = count_zero_values/number_of_values
print(str(percetage*100), 'Of values in PS4 SET are zero')

item_values_count = ps4_data.apply(pandas.value_counts)
# print('Matrix which contains all different sales values for every item from PS4 data \n', item_values_count)

# number of zero sales for every item in every shop and everyday 60*1034 = 62040
number_of_zero_sales = item_values_count.loc[0]
# print('number of zero values for each item:\n',number_of_zero_sales)
print('item with most 0 values:', number_of_zero_sales.idxmax(), 'value:', max(number_of_zero_sales))
print('item with least 0 values:', number_of_zero_sales.idxmin(), 'value:', min(number_of_zero_sales))

plt.plot(number_of_zero_sales.index.tolist(), number_of_zero_sales.tolist())
plt.xlabel('Product id')
plt.ylabel('Number of zero sales')
plt.title('Number of zero sales from PS4 products')
plt.show()

print('------------------------------')


# Category Stats
category_data = pandas.read_csv(INPUT_PATH_P + 'CATEGORIES_ALL_SHOPS' + DATA_TYPE)
category_data = category_data.rename(index=str, columns={'Unnamed: 0': 'category_id'})
category_data = category_data.set_index('category_id')
# drop non category columns
category_data = category_data.drop(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'S_Day'], axis=1)
# print(category_data)

number_of_values = category_data.count().sum()
count_zero_values = category_data[category_data == 0].count().sum()
percetage = count_zero_values/number_of_values
print(str(percetage*100), 'Of values in Category SET are zero')

total_sales_per_cat = category_data.sum()
# print(total_sales_per_cat)

plt.bar(total_sales_per_cat.index.tolist(), total_sales_per_cat.tolist())
plt.xlabel('Category id')
plt.ylabel('Number of total sales')
plt.title('Number of total sales for every Item Category')
plt.show()
