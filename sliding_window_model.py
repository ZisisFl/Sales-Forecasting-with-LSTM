from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from sliding_window_data_gen import DataGenerator
import time

start_time = time.time()  # Time the execution

training_data = [[0 for x in range(973)] for x in range(60)]
validation_data = [[0 for x in range(61)] for x in range(60)]


for i in range(60):
    for j in range(973):
        training_data[i][j] = 'train_id_shop' + str(i) + '_day' + str(j)
    for k in range(61):
        validation_data[i][k] = 'validation_id_shop' + str(i) + '_day' + str(k)

number_of_features = 92  # number_of_features = 215
train_on_n_days = 7

# Parameters
params_train = {'batch_size': 32,  # number of weeks
                'in_dim': (train_on_n_days, number_of_features),
                'out_dim': number_of_features,
                'days_per_shop': 972,
                'shuffle': False}


params_val = {'batch_size': 10,   # number of weeks
              'in_dim': (train_on_n_days, number_of_features),
              'out_dim': number_of_features,
              'days_per_shop': 60,
              'shuffle': False}


# initialize generators
training_generator = DataGenerator(training_data, **params_train)
validation_generator = DataGenerator(validation_data, **params_val)


# design model
model = Sequential()
model.add(LSTM(100, input_shape=(train_on_n_days, number_of_features), return_sequences=False))
model.add(Dense(number_of_features))
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


# evaluate model (val generator, steps (batches of samples) to yield from generator before stopping)
score = model.evaluate_generator(validation_generator, 60, verbose=2)
print(score)

print("--- %s s ---" % (time.time() - start_time))
