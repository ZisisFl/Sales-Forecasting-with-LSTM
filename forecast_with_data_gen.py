from data_generator import DataGenerator
from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from sklearn.preprocessing import MinMaxScaler


train_input = []
test_input = []
train_target = []
test_target = []

for i in range(60):
    train_input.append('train_x_id_' + str(i))
    test_input.append('test_x_id_' + str(i))
    train_target.append('train_y_id_' + str(i))
    test_target.append('test_y_id_' + str(i))


# Parameters
params = {'batch_size': 2,
          'dim': (973, 215),
          'shuffle': False}


# initialize generators
training_generator = DataGenerator(train_input, train_target, **params)
validation_generator = DataGenerator(test_input, test_target, **params)


# design model
model = Sequential()
model.add(LSTM(100, input_shape=(973, 215), return_sequences=True))
model.add(Dense(215))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# fit model
history = model.fit_generator(generator=training_generator,
                              epochs=10,
                              verbose=2,
                              validation_data=validation_generator,
                              use_multiprocessing=False)


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# data is scaled in [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))

# make a prediction
test_pred = model.predict_generator(validation_generator)

# inverse data scaling before applying rmse
test_pred_inv = scaler.inverse_transform(test_pred)
#test_y_inv = scaler.inverse_transform(test_y)

#rmse = sqrt(mean_squared_error(test_y_inv, test_pred_inv))
#print('Val RMSE: %.3f' % rmse)
