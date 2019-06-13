import pandas
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time

start_time = time.time()  # Time the execution


DATA_TYPE = '.csv'
INPUT_PATH = 'data/processed/'
INPUT_DATA = 'PS4_SET'

dataframe = pandas.read_csv(INPUT_PATH + INPUT_DATA + DATA_TYPE)
dataframe = dataframe.rename(index=str, columns={'Unnamed: 0': 'item_id'})
dataframe = dataframe.set_index('item_id')
x_data = dataframe

# make the problem supervised test_df is the train_df shifted -1
y_data = x_data.shift(-1)
y_data.fillna(0, inplace=True)

# scale data in [-1, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
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
model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2]), stateful=False))
model.add(Dense(206))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


# create checkpoint callback to keep epoch with smallest val_loss
filepath = 'h5_files/simple_model_best.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# early stoping callback to stop training after a given number of epochs
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)


# fit model
history = model.fit(train_x, train_y, epochs=10,
                    validation_data=(test_x, test_y), verbose=2,
                    callbacks=[checkpoint, early_stopping_callback], shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# make a prediction
test_pred = model.predict(test_x)


# inverse data scaling before applying rmse
test_pred = scaler.inverse_transform(test_pred)
test_y_inv = scaler.inverse_transform(test_y)

# calculate rmse of predicted and test
rmse = sqrt(mean_squared_error(test_y_inv, test_pred))
print('Val RMSE: %.3f' % rmse)

print("--- %s s ---" % (time.time() - start_time))
