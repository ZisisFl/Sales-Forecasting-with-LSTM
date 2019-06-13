import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler

DATA_TYPE = '.csv'
INPUT_PATH = 'data/processed/'
INPUT_DATA = 'PS4_SET_ALL_SHOPS'
# INPUT_DATA = 'CATEGORIES_ALL_SHOPS'

FLAG = 1  # 1 GENERATE DATA PER DAY / 0 GENERATE DATA PER SHOP
INCLUDE_SUNDAYS = 0  # 1 NO / 0 YES

dataframe = pandas.read_csv(INPUT_PATH + INPUT_DATA + DATA_TYPE)
dataframe = dataframe.rename(index=str, columns={'Unnamed: 0': 'item_id'})
dataframe = dataframe.set_index('item_id')
x_data = dataframe

# make the problem supervised test_df is the train_df shifted -1
y_data = x_data.shift(-1)
y_data.fillna(0, inplace=True)

# take the 2 last months for test
test_date_range = pandas.date_range(start='2015/09/01', end='2015/10/31')
test_n_days = test_date_range.nunique()
n_shops = 60

# change dimensions in order to prepare data
x_data = x_data.T
y_data = y_data.T

# initialize empty test_x and test_y data frames
test_x = pandas.DataFrame(index=range(0))
test_y = pandas.DataFrame(index=range(0))

for i in range(n_shops):
    for j in range(test_n_days):
        date = str(pandas.to_datetime(test_date_range[j]).date())

        df1 = x_data[date]
        df1 = df1.iloc[:, i:i+1]
        test_x = pandas.concat([test_x, df1], axis=1)  # ,sort=True

        df2 = y_data[date]
        df2 = df2.iloc[:, i:i+1]
        test_y = pandas.concat([test_y, df2], axis=1)  # ,sort=True


# drop the test part from the x_data and y_data data frames
for j in range(test_n_days):
    x_data = x_data.drop([str(pandas.to_datetime(test_date_range[j]).date())], axis=1)
    y_data = y_data.drop([str(pandas.to_datetime(test_date_range[j]).date())], axis=1)

# x_data and y_data contain whole data minus test
train_x = x_data
train_y = y_data

# return data to original dimensions
train_x = train_x.T
test_x = test_x.T
train_y = train_y.T
test_y = test_y.T


if FLAG == 0:
    # pad test_x and test_y with zeros to match train shape
    # pad(array, ((top, bottom), (left, right)), mode)
    test_x = np.pad(test_x, ((train_x.shape[0] - test_x.shape[0], 0), (0, 0)), 'constant', constant_values=0)
    test_y = np.pad(test_y, ((train_y.shape[0] - test_y.shape[0], 0), (0, 0)), 'constant', constant_values=0)
    # print(train_x.shape[0]/n_shops, test_x.shape[0]/n_shops)

    # scale data in [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    train_y = scaler.fit_transform(train_y)
    test_y = scaler.fit_transform(test_y)

    # shape data for lstm model (Samples, Time steps, Features)
    # (60 shops, 973 days, 207 items + 7 onehot days + 1 s_day) for forecast with data gen
    train_x = train_x.reshape((n_shops, 1034 - test_n_days, train_x.shape[1]))
    train_y = train_y.reshape((n_shops, 1034 - test_n_days, train_y.shape[1]))
    test_x = test_x.reshape((n_shops, 1034 - test_n_days, test_x.shape[1]))
    test_y = test_y.reshape((n_shops, 1034 - test_n_days, test_y.shape[1]))

    # create npy files for forecast_with_data_gen.py
    i = 0
    for row in train_x:
        np.save('data/x_data/' + 'train_x' + '_id_' + str(i), row)
        i = i + 1

    i = 0
    for row in train_y:
        np.save('data/y_data/' + 'train_y' + '_id_' + str(i), row)
        i = i + 1

    i = 0
    for row in test_x:
        np.save('data/x_data/' + 'test_x' + '_id_' + str(i), row)
        i = i + 1

    i = 0
    for row in test_y:
        np.save('data/y_data/' + 'test_y' + '_id_' + str(i), row)
        i = i + 1
else:
    if INCLUDE_SUNDAYS == 1:
        # remove sundays from datasets
        train_x = train_x[train_x.Sun != 1]
        test_x = test_x[test_x.Sun != 1]
        train_x = train_x.drop(['Sun'], axis=1)
        test_x = test_x.drop(['Sun'], axis=1)
        # print(train_x)
        # print(test_x)
        # print(train_x.shape[0]/n_shops, test_x.shape[0]/n_shops)

    # scale data in [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    train_y = scaler.fit_transform(train_y)
    test_y = scaler.fit_transform(test_y)

    # create npy files for sliding_window_model
    i = 0
    j = 0
    for row in train_x:
        if i == int(train_x.shape[0] / n_shops):
            i = 0
            j = j + 1
        np.save('data/data_rows/' + 'train_id_shop' + str(j) + '_day' + str(i), row)
        i = i + 1

    i = 0
    j = 0
    for row in test_x:
        if i == int(test_x.shape[0] / n_shops):
            i = 0
            j = j + 1
        np.save('data/data_rows/' + 'validation_id_shop' + str(j) + '_day' + str(i), row)
        i = i + 1
