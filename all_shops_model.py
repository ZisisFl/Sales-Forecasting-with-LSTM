import pandas
from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from sklearn.preprocessing import MinMaxScaler

DATA_TYPE = '.csv'
INPUT_PATH = 'data/processed/'
INPUT_DATA = 'PS4_SET_ALL_SHOPS'

dataframe = pandas.read_csv(INPUT_PATH + INPUT_DATA + DATA_TYPE)
dataframe = dataframe.rename(index=str, columns={'Unnamed: 0': 'item_id'})
dataframe = dataframe.set_index('item_id')
train_df = dataframe

# make the problem supervised test_df is the train_df shifted -1
test_df = train_df
test_df = test_df.shift(-1)
test_df.fillna(0, inplace=True)

# take the 2 last months for test
date_range = pandas.date_range(start='2015/09/01', end='2015/10/31')
n_days = date_range.nunique()
n_shops = 60

# change dimensions in order to prepare data
train_df = train_df.T
test_df = test_df.T

test_x = pandas.DataFrame(index=range(215))
test_y = pandas.DataFrame(index=range(215))

for i in range(n_shops):
    for j in range(n_days):
        date = str(pandas.to_datetime(date_range[j]).date())

        df1 = train_df[date]
        df1 = df1.iloc[:, i:i+1]
        test_x = pandas.concat([test_x, df1], axis=1)

        df2 = test_df[date]
        df2 = df2.iloc[:, i:i+1]
        test_y = pandas.concat([test_y, df2], axis=1)

for j in range(n_days):
    test_df = test_df.drop([str(pandas.to_datetime(date_range[j]).date())], axis=1)
    train_df = train_df.drop([str(pandas.to_datetime(date_range[j]).date())], axis=1)

train_x = train_df
train_y = test_df
test_x = test_x.dropna()
test_y = test_y.dropna()

# return data to original dimensions
train_x = train_x.T
test_x = test_x.T
train_y = train_y.T
test_y = test_y.T

# scale data in [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)
train_y = scaler.fit_transform(train_y)
test_y = scaler.fit_transform(test_y)

# shape data for lstm model
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# design model
model = Sequential()
model.add(LSTM(100, batch_input_shape=(60, train_x.shape[1], train_x.shape[2]), stateful=True))
model.add(Dense(215))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# fit model
history = model.fit(train_x, train_y, epochs=10, batch_size=60,   # 58380 and 3660 have same max divisor 60
                    validation_data=(test_x, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
y_pre = model.predict(test_x, 60) # batch size = 60

rmse = sqrt(mean_squared_error(test_y, y_pre))
print('Val RMSE: %.3f' % rmse)
