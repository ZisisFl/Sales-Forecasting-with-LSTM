import pandas

DATA_TYPE = '.csv'
INPUT_PATH = 'data/raw/'
OUTPUT_PATH = 'data/processed/'

dataframe = pandas.read_csv(INPUT_PATH + 'sales_train_v2' + DATA_TYPE)

# remove rows with items value < 0 these items are returned to the store
dataframe = dataframe[dataframe['item_cnt_day'] > 0]

# multiply item_price and item_cnt_day to create sales column
dataframe['sales'] = dataframe.apply(lambda row: row['item_price'] * row['item_cnt_day'], axis=1)
dataframe = dataframe.drop(['item_price', 'item_cnt_day'], axis=1)

# group final dataset
dataframe = dataframe.groupby(['date_block_num', 'shop_id', 'item_id']).sum()['sales']

print(dataframe)

dataframe.to_csv(OUTPUT_PATH + 'train_data' + DATA_TYPE, encoding='utf-8', index=True, header=True)