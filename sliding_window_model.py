from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from sklearn.preprocessing import MinMaxScaler
from random import randint
from sliding_window_data_gen import DataGenerator

training_data = [[0 for x in range(973)] for x in range(60)]
validation_data = [[0 for x in range(61)] for x in range(60)]

for i in range(60):
    for j in range(973):
        training_data[i][j] = 'train_id_shop' + str(i) + '_day' + str(j)
    for k in range(61):
        validation_data[i][k] = 'validation_id_shop' + str(i) + '_day' + str(k)

#print(training_data)
#print(validation_data)


# Parameters
params_train = {'batch_size': 32,
                'in_dim': (7, 215),
                'out_dim': 215,
                'days_per_shop': 972,
                'shuffle': False}


params_val = {'batch_size': 10,
              'in_dim': (7, 215),
              'out_dim': 215,
              'days_per_shop': 60,
              'shuffle': False}

# initialize generators
training_generator = DataGenerator(training_data, **params_train)
validation_generator = DataGenerator(validation_data, **params_val)

# design model
model = Sequential()
model.add(LSTM(100, input_shape=(7, 215), return_sequences=False))
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

# make a prediction
test_pred = model.predict_generator(validation_generator)
# evaluate model
score = model.evaluate_generator(validation_generator, 10, verbose=2)
print(score)
