import pandas
import numpy as np


DATA_TYPE = '.csv'
INPUT_PATH = 'data/processed/'
# INPUT_DATA = 'CATEGORIES_ALL_SHOPS'
INPUT_DATA = 'PS4_SET_ALL_SHOPS'

FLAG = 1  # 0 GENERATE DATA PER SHOP / 1 GENERATE DATA PER DAY
SCALE_DATA = 1  # 0 NO SCALING / 1 SCALING

dataframe = pandas.read_csv(INPUT_PATH + INPUT_DATA + DATA_TYPE)
dataframe = dataframe.rename(index=str, columns={'Unnamed: 0': 'item_id'})
dataframe = dataframe.set_index('item_id')
x_data = dataframe

# make the problem supervised test_df is the train_df shifted -1
y_data = x_data.shift(-1)
y_data.fillna(0, inplace=True)

# drop S_day [:, :-1] or one hot rep and S_day [:, :-8] from training
x_data = x_data.iloc[:, :-8]

# drop last 8 columns referring to one hot rep and S_day from the target values
y_data = y_data.iloc[:, :-8]

test_date_range = pandas.date_range(start='2015/09/01', end='2015/10/31')

n_shops = 60  # number of shops
days_per_shop = 1034  # number of days both train and test
test_days = test_date_range.nunique()  # number of test days
train_days = days_per_shop - test_days  # number of train days
n_features = 207  # 207 for products / 84 for categories

x_data = x_data.values
y_data = y_data.values

# print(np.count_nonzero(x_data))

train_x = np.empty([0, n_features])
train_y = np.empty([0, n_features])
test_x = np.empty([0, n_features])
test_y = np.empty([0, n_features])


def scaler(target_data):
    minmax = []
    n_columns = target_data.shape[1]
    n_rows = target_data.shape[0]
    print(n_rows, n_columns)
    for column in range(n_columns):
        value_min = min(target_data[:, column])
        value_max = max(target_data[:, column])
        minmax.append([value_min, value_max])
    return minmax


def scale_data(target_data, minmax):
    target_data = target_data.astype(float)
    n_columns = target_data.shape[1]
    n_rows = target_data.shape[0]
    for column in range(n_columns):
        for row in range(n_rows):
            if minmax[column][1] > 0:  # if max = 0 there is division error and column is for sure full of zero
                target_data[row, column] = (target_data[row, column] - minmax[column][0]) / (minmax[column][1] - minmax[column][0])
    return target_data


if FLAG == 0:
    for i in range(0, n_shops):
        np_array = x_data[(days_per_shop - test_days) + (i * days_per_shop): days_per_shop + (i * days_per_shop), :]
        # pad test_x and test_y with zeros to match train shape
        # pad(array, ((top, bottom), (left, right)), mode)
        np_array = np.pad(np_array, ((973 - 61, 0), (0, 0)), 'constant', constant_values=0)
        test_x = np.concatenate((test_x, np_array), axis=0)

        np_array = y_data[(days_per_shop - test_days) + (i * days_per_shop): days_per_shop + (i * days_per_shop), :]
        np_array = np.pad(np_array, ((973 - 61, 0), (0, 0)), 'constant', constant_values=0)
        test_y = np.concatenate((test_y, np_array), axis=0)

        np_array = x_data[i * days_per_shop:(days_per_shop - test_days) + (i * days_per_shop), :]
        train_x = np.concatenate((train_x, np_array), axis=0)

        np_array = x_data[i * days_per_shop:(days_per_shop - test_days) + (i * days_per_shop), :]
        train_y = np.concatenate((train_y, np_array), axis=0)

    if SCALE_DATA == 1:
        scaler_train_x = scaler(train_x)
        scaler_train_y = scaler(train_y)
        scaler_test_x = scaler(test_x)
        scaler_test_y = scaler(test_y)

        train_x = scale_data(train_x, scaler_train_x)
        train_y = scale_data(train_y, scaler_train_y)
        test_x = scale_data(test_x, scaler_test_x)
        test_y = scale_data(test_y, scaler_test_y)

        # save test_y scaling matrix in order to inverse scaling
        np.save('data/y_data/test_y_minmax', scaler_test_y)

    # shape data for lstm model (Samples, Time steps, Features)
    train_x = train_x.reshape((n_shops, 1034 - test_days, train_x.shape[1]))  # 60, 973, 215
    train_y = train_y.reshape((n_shops, 1034 - test_days, train_y.shape[1]))
    test_x = test_x.reshape((n_shops, 1034 - test_days, test_x.shape[1]))
    test_y = test_y.reshape((n_shops, 1034 - test_days, test_y.shape[1]))

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
    for i in range(0, n_shops):
        np_array = x_data[(days_per_shop - test_days) + (i * days_per_shop): days_per_shop + (i * days_per_shop), :]
        test_x = np.concatenate((test_x, np_array), axis=0)

        np_array = y_data[(days_per_shop - test_days) + (i * days_per_shop): days_per_shop + (i * days_per_shop), :]
        test_y = np.concatenate((test_y, np_array), axis=0)

        np_array = x_data[i * days_per_shop:(days_per_shop - test_days) + (i * days_per_shop), :]
        train_x = np.concatenate((train_x, np_array), axis=0)

        np_array = y_data[i * days_per_shop:(days_per_shop - test_days) + (i * days_per_shop), :]
        train_y = np.concatenate((train_y, np_array), axis=0)

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
