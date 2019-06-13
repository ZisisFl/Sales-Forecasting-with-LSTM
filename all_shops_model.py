import pandas
import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
        test_x = pandas.concat([test_x, df1], axis=1)

        df2 = y_data[date]
        df2 = df2.iloc[:, i:i+1]
        test_y = pandas.concat([test_y, df2], axis=1)

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

# pad test_x and test_y with zeros to match train shape
# pad(array, ((top, bottom), (left, right)), mode)
test_x = np.pad(test_x, ((train_x.shape[0]-test_x.shape[0], 0), (0, 0)), 'constant', constant_values=0)
test_y = np.pad(test_y, ((train_y.shape[0]-test_y.shape[0], 0), (0, 0)), 'constant', constant_values=0)

# scale data in [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)
train_y = scaler.fit_transform(train_y)
test_y = scaler.fit_transform(test_y)

# shape data for lstm model (Samples, Time steps, Features) (60 shops, 973 days, 207 items + 7 onehot days + 1 s_day)
train_x = train_x.reshape((n_shops, 1034 - test_n_days, train_x.shape[1]))  # 60, 973, 215
train_y = train_y.reshape((n_shops, 1034 - test_n_days, train_y.shape[1]))
test_x = test_x.reshape((n_shops, 1034 - test_n_days, test_x.shape[1]))
test_y = test_y.reshape((n_shops, 1034 - test_n_days, test_y.shape[1]))


# design model
model = Sequential()
model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dense(train_x.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


# create checkpoint callback to keep epoch with smallest val_loss
filepath = 'h5_files/all_shops_model_best.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# early stoping callback to stop training after a given number of epochs
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)


# fit model
history = model.fit(train_x, train_y,
                    epochs=10,
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


# make a prediction
test_pred = model.predict(test_x)


# inverse data scaling before applying rmse
test_pred_inv = np.empty([n_shops, 1034 - test_n_days, train_x.shape[2]])
test_y_inv = np.empty([n_shops, 1034 - test_n_days, train_x.shape[2]])


# invert scaling for for test_pred and test_y
for i in range(n_shops):
    test_pred_inv[i, ] = scaler.inverse_transform(test_pred[i])
    test_y_inv[i, ] = scaler.inverse_transform(test_y[i])


# calculate rmse for each shop
rmse = []
for i in range(n_shops):
    rmse.append(sqrt(mean_squared_error(test_y_inv[i], test_pred_inv[i])))
print(rmse)


print('Val RMSE: %.3f' % (sum(rmse)/len(rmse)))

print("--- %s s ---" % (time.time() - start_time))
