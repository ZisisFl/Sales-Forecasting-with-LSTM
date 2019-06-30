import pandas
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
import numpy as np
from sklearn.metrics import mean_squared_error
from data_generator import DataGenerator
from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
import time

DATA_SCALED = 1  # 0  NON SCALED DATA / 1 SCALED DATA

start_time = time.time()  # Time the execution

# create lists with file indexes
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
training_params = {'batch_size': 2,
                   'dim': (973, 207),
                   'shuffle': False}

validation_params = {'batch_size': 2,
                     'dim': (973, 207),
                     'shuffle': False}


# initialize generators
training_generator = DataGenerator(train_input, train_target, **training_params)
validation_generator = DataGenerator(test_input, test_target, **validation_params)


# design model
model = Sequential()
model.add(LSTM(100, input_shape=(973, 207), return_sequences=True))
model.add(Dense(207))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


# create checkpoint callback to keep epoch with smallest val_loss
filepath = 'h5_files/forecast_with_data_gen_best.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# early stoping callback to stop training after a given number of epochs
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)


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

# use the best forecast
model = load_model('h5_files/forecast_with_data_gen_best.h5')


# make a prediction
test_pred = model.predict_generator(validation_generator)


def inverse_scaling(target_data, minmax_matrix):
    n_columns = target_data.shape[1]
    n_rows = target_data.shape[0]
    for column in range(n_columns):
        for row in range(n_rows):
            target_data[row, column] = (target_data[row, column] * (minmax_matrix[column][1] - minmax_matrix[column][0])) + minmax_matrix[column][0]
    return target_data


# initialize test_y matrix
test_y = np.empty([0, 207])
for i in range(60):
    np_array = np.load('data/y_data/' + test_target[i] + '.npy')
    test_y = np.concatenate((test_y, np_array), axis=0)


# if data are scaled load the mimax matrix to inverse data
if DATA_SCALED == 1:
    scaler = np.load('data/y_data/test_y_minmax.npy')
    test_y = inverse_scaling(test_y, scaler)

    # reshape data to 58380, features to inverse scaling
    test_pred = test_pred.reshape(test_pred.shape[0]*test_pred.shape[1], test_pred.shape[2])

    # inverse scaling
    test_pred = inverse_scaling(test_pred, scaler)

    # reshape data to 60, 973, features to calculate MSE for every shop
    test_pred = test_pred.reshape(60, 973, test_pred.shape[1])


# reshape test_y data to match test_pred shape
test_y = test_y.reshape(60, 973, test_y.shape[1])

# calculate MSE for each shop
mse = []
for i in range(60):
    mse.append((mean_squared_error(test_y[i, -61:], test_pred[i, -61:])))

print('MSE of all shops:', mse)
print('MSE of Shop 25: %.3f' % mse[25])


# print the average MSE of all shops
print('Val MSE: %.3f' % (sum(mse)/len(mse)))


# evaluate model (val generator, steps (batches of samples) to yield from generator before stopping)
# score = model.evaluate_generator(validation_generator, 60, verbose=2)
# print('Mean squared error:', score[0])

# create plot with target everyday sales from all products vs predicted sales
test_y_list = []
test_pred_list = []
test_date_range = pandas.date_range(start='2015/09/01', end='2015/10/31')


for k in range(test_y.shape[0]):
    for i in range(test_y.shape[1]-61, test_y.shape[1]):
        sum_per_day_y = 0
        sum_per_day_pred = 0
        for j in range(test_y.shape[2]):
            sum_per_day_y = sum_per_day_y + test_y[k, i, j]
            sum_per_day_pred = sum_per_day_pred + test_pred[k, i, j]
        test_y_list.append(sum_per_day_y)
        test_pred_list.append(sum_per_day_pred)

plot_shop = 25
pyplot.plot(test_date_range, test_y_list[plot_shop*61:(plot_shop*61)+61], label='target')
pyplot.plot(test_date_range, test_pred_list[plot_shop*61:(plot_shop*61)+61], label='predicted')
pyplot.xticks(rotation=45)
pyplot.legend()
pyplot.show()

print("--- %s s ---" % (time.time() - start_time))
