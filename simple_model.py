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
train_df = dataframe.T

# make the problem supervised y is the x shifted -1
test_df = train_df
test_df = test_df.shift(-1)
test_df.fillna(0, inplace=True)

# scale data in [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_df = scaler.fit_transform(train_df)
test_df = scaler.fit_transform(test_df)

# model trains on 32 months and tests on 2 months
train_x, train_y = train_df[0:-61], test_df[0:-61]
test_x, test_y = train_df[-61:], test_df[-61:]

# shape data for lstm model
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# design model
model = Sequential()
model.add(LSTM(100, batch_input_shape=(1, train_x.shape[1], train_x.shape[2]), stateful=True))
model.add(Dense(206))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

# fit model
history = model.fit(train_x, train_y, epochs=10, batch_size=1,
                    validation_data=(test_x, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
y_pre = model.predict(test_x, 1)

rmse = sqrt(mean_squared_error(test_y, y_pre))
print('Val RMSE: %.3f' % rmse)