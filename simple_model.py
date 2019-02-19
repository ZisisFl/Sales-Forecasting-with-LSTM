import pandas
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


DATA_TYPE = '.csv'
INPUT_PATH = 'data/processed/'
INPUT_DATA = 'PS4_SET'

dataframe = pandas.read_csv(INPUT_PATH + INPUT_DATA + DATA_TYPE)
dataframe = dataframe.rename(index=str, columns={'Unnamed: 0': 'item_id'})
dataframe = dataframe.set_index('item_id')
x_data = dataframe.T

# make the problem supervised test_df is the train_df shifted -1
y_data = x_data.shift(-1)
y_data.fillna(0, inplace=True)

# scale data in [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
x_data = scaler.fit_transform(x_data)
y_data = scaler.fit_transform(y_data)

# model trains over 32 months and tests on 2 months
train_x, train_y = x_data[0:-61], y_data[0:-61]
test_x, test_y = x_data[-61:], y_data[-61:]

# shape data for LSTM model
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# design model
model = Sequential()
model.add(LSTM(100, batch_input_shape=(1, train_x.shape[1], train_x.shape[2]), stateful=True))
model.add(Dense(206))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# fit model
history = model.fit(train_x, train_y, epochs=10, batch_size=1,   # 973 dividers = 1 7 139 973 61 dividers = 1 61
                    validation_data=(test_x, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# make a prediction
test_pred = model.predict(test_x, 1)


# inverse data scaling before applying rmse
test_pred = scaler.inverse_transform(test_pred)
test_y_inv = scaler.inverse_transform(test_y)

# calculate rmse of predicted and test
rmse = sqrt(mean_squared_error(test_y_inv, test_pred))
print('Val RMSE: %.3f' % rmse)