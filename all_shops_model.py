import pandas
import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time

start_time = time.time()  # Time the execution

DATA_TYPE = '.csv'
INPUT_PATH = 'data/processed/'
# INPUT_DATA = 'CATEGORIES_ALL_SHOPS'
INPUT_DATA = 'PS4_SET_ALL_SHOPS'

SCALE_DATA = 0  # 0 NO SCALING / 1 SCALING

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


for i in range(0, n_shops):
    np_array = x_data[(days_per_shop-test_days) + (i * days_per_shop): days_per_shop + (i * days_per_shop), :]
    # pad test_x and test_y with zeros to match train shape
    # pad(array, ((top, bottom), (left, right)), mode)
    np_array = np.pad(np_array, ((train_days - test_days, 0), (0, 0)), 'constant', constant_values=0)
    test_x = np.concatenate((test_x, np_array), axis=0)

    np_array = y_data[(days_per_shop - test_days) + (i * days_per_shop): days_per_shop + (i * days_per_shop), :]
    np_array = np.pad(np_array, ((train_days - test_days, 0), (0, 0)), 'constant', constant_values=0)
    test_y = np.concatenate((test_y, np_array), axis=0)

    np_array = x_data[i * days_per_shop:(days_per_shop - test_days) + (i * days_per_shop), :]
    train_x = np.concatenate((train_x, np_array), axis=0)

    np_array = x_data[i * days_per_shop:(days_per_shop - test_days) + (i * days_per_shop), :]
    train_y = np.concatenate((train_y, np_array), axis=0)


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


def inverse_scaling(target_data, minmax_matrix):
    n_columns = target_data.shape[1]
    n_rows = target_data.shape[0]
    for column in range(n_columns):
        for row in range(n_rows):
            target_data[row, column] = (target_data[row, column] * (minmax_matrix[column][1] - minmax_matrix[column][0])) + minmax_matrix[column][0]
    return target_data


if SCALE_DATA == 1:
    scaler_train_x = scaler(train_x)
    scaler_train_y = scaler(train_y)
    scaler_test_x = scaler(test_x)
    scaler_test_y = scaler(test_y)

    train_x = scale_data(train_x, scaler_train_x)
    train_y = scale_data(train_y, scaler_train_y)
    test_x = scale_data(test_x, scaler_test_x)
    test_y = scale_data(test_y, scaler_test_y)

    #scaler_x = MinMaxScaler(feature_range=(0, 1))
    #scaler_y = MinMaxScaler(feature_range=(0, 1))
    #train_x = scaler_x.fit_transform(train_x)
    #train_y = scaler_y.fit_transform(train_y)
    #test_x = scaler_x.fit_transform(test_x)
    #test_y = scaler_y.fit_transform(test_y)


# shape data for lstm model (Samples, Time steps, Features) (60 shops, 973 days, 207 items + 7 onehot days + 1 s_day)
train_x = train_x.reshape((n_shops, 1034 - test_days, train_x.shape[1]))  # 60, 973, 215
train_y = train_y.reshape((n_shops, 1034 - test_days, train_y.shape[1]))
test_x = test_x.reshape((n_shops, 1034 - test_days, test_x.shape[1]))
test_y = test_y.reshape((n_shops, 1034 - test_days, test_y.shape[1]))

# print(np.count_nonzero(test_y[25, :, :]))


# design model
model = Sequential()
model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dense(train_y.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


# create checkpoint callback to keep epoch with smallest val_loss
filepath = 'h5_files/all_shops_model_best.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# early stoping callback to stop training after a given number of epochs
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)


# fit model
history = model.fit(train_x, train_y,
                    epochs=100,
                    batch_size=2,
                    validation_data=(test_x, test_y),
                    verbose=2,
                    callbacks=[checkpoint, early_stopping_callback],
                    shuffle=False)


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# use the best forecast
model = load_model('h5_files/all_shops_model_best.h5')


# make a prediction
test_pred = model.predict(test_x)


# invert scaling for for test_pred and test_y
if SCALE_DATA == 1:
    # reshape data to 58380, features to inverse scaling
    test_y = test_y.reshape(train_y.shape[0] * train_y.shape[1], train_y.shape[2])
    test_pred = test_pred.reshape(train_y.shape[0] * train_y.shape[1], train_y.shape[2])

    test_pred = inverse_scaling(test_pred, scaler_test_y)
    test_y = inverse_scaling(test_y, scaler_test_y)

    # reshape data to 60, 973, features to calculate MSE for every shop
    test_y = test_y.reshape(train_y.shape[0], train_y.shape[1], train_y.shape[2])
    test_pred = test_pred.reshape(train_y.shape[0], train_y.shape[1], train_y.shape[2])

    #for i in range(60):
    #    test_pred[i, ] = scaler_y.inverse_transform(test_pred[i])
    #    test_y[i, ] = scaler_y.inverse_transform(test_y[i])


# calculate MSE for each shop
mse = []
for i in range(n_shops):
    mse.append((mean_squared_error(test_y[i, -61:], test_pred[i, -61:])))

print('MSE of all shops:', mse)
print('MSE of Shop 25: %.3f' % mse[25])


# print the average MSE of all shops
print('Val MSE: %.3f' % (sum(mse)/len(mse)))


# create plot with target everyday sales from all products vs predicted sales
test_y_list = []
test_pred_list = []
test_date_range = pandas.date_range(start='2015/09/01', end='2015/10/31')


for k in range(test_y.shape[0]):
    for i in range(test_y.shape[1]-61, test_y.shape[1]):
        sum_per_day_y = 0
        sum_per_day_pred = 0
        for j in range(test_y.shape[2]):
            sum_per_day_y = sum_per_day_y + test_y[k, i, j]
            sum_per_day_pred = sum_per_day_pred + test_pred[k, i, j]
        test_y_list.append(sum_per_day_y)
        test_pred_list.append(sum_per_day_pred)

plot_shop = 25
pyplot.plot(test_date_range, test_y_list[plot_shop*61:(plot_shop*61)+61], label='target')
pyplot.plot(test_date_range, test_pred_list[plot_shop*61:(plot_shop*61)+61], label='predicted')
pyplot.xticks(rotation=45)
pyplot.legend()
pyplot.show()

print("--- %s s ---" % (time.time() - start_time))
