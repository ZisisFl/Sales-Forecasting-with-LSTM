from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Masking
from matplotlib import pyplot
from sliding_window_data_gen import DataGenerator
import time
import numpy as np

start_time = time.time()  # Time the execution

training_data = [[0 for x in range(973)] for x in range(60)]
validation_data = [[0 for x in range(61)] for x in range(60)]


for i in range(60):
    for j in range(973):
        training_data[i][j] = 'train_id_shop' + str(i) + '_day' + str(j)
    for k in range(61):
        validation_data[i][k] = 'validation_id_shop' + str(i) + '_day' + str(k)

# number_of_features = 92  # for categories
number_of_features = 207  # for products
train_on_n_days = 7

# Parameters
params_train = {'batch_size': 32,  # number of weeks
                'in_dim': (train_on_n_days, number_of_features),
                'out_dim': number_of_features,
                'days_per_shop': 972,
                'data_type': 'train',
                'shuffle': False}


params_val = {'batch_size': 10,   # number of weeks
              'in_dim': (train_on_n_days, number_of_features),
              'out_dim': number_of_features,
              'days_per_shop': 60,
              'data_type': 'test',
              'shuffle': False}


# initialize generators
training_generator = DataGenerator(training_data, **params_train)
validation_generator = DataGenerator(validation_data, **params_val)


# design model
model = Sequential()
model.add(LSTM(100, input_shape=(train_on_n_days, number_of_features), return_sequences=False))
model.add(Dense(number_of_features))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


# create checkpoint callback to keep epoch with smallest val_loss
filepath = 'h5_files/sliding_window_best.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# early stoping callback to stop training after a given number of epochs
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)


# fit model
history = model.fit_generator(generator=training_generator,
                              epochs=100,
                              verbose=2,
                              validation_data=validation_generator,
                              callbacks=[checkpoint, early_stopping_callback],
                              use_multiprocessing=False)


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
test_pred = model.predict_generator(validation_generator)
print(test_pred.shape)
test_pred = test_pred.reshape(test_pred.shape[0]//10, 10, test_pred.shape[1])

test_y_list = []
test_pred_list = []
x_labels = []

for k in range(10):
    x_labels.append('Day' + str(k))

for i in range(test_pred.shape[0]):
    for j in range(test_pred.shape[1]):
        temp = np.load('data/data_rows_target/' + 'shop_' + str(i) + '_target' + '.npy')
        sum_per_day_y = 0
        sum_per_day_pred = 0
        for n in range(test_pred.shape[2]):
            sum_per_day_y = sum_per_day_y + temp[j, n]
            sum_per_day_pred = sum_per_day_pred + test_pred[i, j, n]
        test_y_list.append(sum_per_day_y)
        test_pred_list.append(sum_per_day_pred)


plot_shop = 25

pyplot.plot(x_labels, test_y_list[plot_shop*10:(plot_shop*10)+10], label='target')
pyplot.plot(x_labels, test_pred_list[plot_shop*10:(plot_shop*10)+10], label='predicted')
pyplot.xticks(rotation=45)
pyplot.legend()
pyplot.show()

# evaluate model (val generator, steps (batches of samples) to yield from generator before stopping)
score = model.evaluate_generator(validation_generator, 1000, verbose=2)
print(score)

print("--- %s s ---" % (time.time() - start_time))
