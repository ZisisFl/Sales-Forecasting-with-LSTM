from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from sliding_window_data_gen import DataGenerator
import time

start_time = time.time()  # Time the execution

training_data = [[0 for x in range(834)] for x in range(60)]
validation_data = [[0 for x in range(53)] for x in range(60)]


for i in range(60):
    for j in range(834):
        training_data[i][j] = 'train_id_shop' + str(i) + '_day' + str(j)
    for k in range(53):
        validation_data[i][k] = 'validation_id_shop' + str(i) + '_day' + str(k)


# Parameters
params_train = {'batch_size': 32,
                'in_dim': (7, 214),
                'out_dim': 214,
                'days_per_shop': 833,
                'shuffle': False}


params_val = {'batch_size': 10,
              'in_dim': (7, 214),
              'out_dim': 214,
              'days_per_shop': 52,
              'shuffle': False}


# initialize generators
training_generator = DataGenerator(training_data, **params_train)
validation_generator = DataGenerator(validation_data, **params_val)


# design model
model = Sequential()
model.add(LSTM(100, input_shape=(7, 214), return_sequences=False))
model.add(Dense(214))
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
score = model.evaluate_generator(validation_generator, 60, verbose=2)
print(score)

print("--- %s s ---" % (time.time() - start_time))