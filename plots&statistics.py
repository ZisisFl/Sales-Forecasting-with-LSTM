import pandas
import numpy as np
from matplotlib import pyplot as plt

DATA_TYPE = '.csv'
INPUT_PATH = 'data/processed/'
INPUT_DATA = 'PS4_SET_ALL_SHOPS'

# General numbers from all data
sales_data = pandas.read_csv('data/raw/sales_train_v2.csv')
print('General number from all data \n------------------------------')
print('number of shops:', sales_data['shop_id'].nunique())
print('number of items:', sales_data['item_id'].nunique())
print('number of months:', sales_data['date_block_num'].nunique())

category_data = pandas.read_csv('data/raw/item_categories.csv')
print('number of categories:', category_data['item_category_name'].nunique())

date_range = pandas.date_range(start='1/1/2013', end='31/10/2015')
print('number of days', date_range.shape[0])
print('------------------------------')

# PS4 Stats
dataframe = pandas.read_csv(INPUT_PATH + INPUT_DATA + DATA_TYPE)
dataframe = dataframe.rename(index=str, columns={'Unnamed: 0': 'item_id'})
dataframe = dataframe.set_index('item_id')
# drop non item columns
dataframe = dataframe.drop(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'S_Day'], axis=1)
# print(dataframe)

item_values_count = dataframe.apply(pandas.value_counts)
print('Matrix which contains all different sales values for every item from PS4 data \n', item_values_count)

# number of zero sales for every item in every shop and everyday 60*1034 = 62040
number_of_zero_sales = item_values_count.loc[0]
print('number of zero values for each item:\n',number_of_zero_sales)
print('item with most 0 values:', number_of_zero_sales.idxmax(), 'value:', max(number_of_zero_sales))
print('item with least 0 values:', number_of_zero_sales.idxmin(), 'value:', min(number_of_zero_sales))


plt.plot(number_of_zero_sales.index.tolist(), number_of_zero_sales.tolist())
plt.xlabel('Product id')
plt.ylabel('Number of zero sales')
plt.title('Number of zero sales from PS4 products')
plt.show()




