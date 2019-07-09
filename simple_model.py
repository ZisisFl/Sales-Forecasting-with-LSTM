import pandas
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time

start_time = time.time()  # Time the execution


DATA_TYPE = '.csv'
INPUT_PATH = 'data/processed/'
INPUT_DATA = 'PS4_SET'

SCALE_DATA = 1  # 0 NO SCALING / 1 SCALING

dataframe = pandas.read_csv(INPUT_PATH + INPUT_DATA + DATA_TYPE)
dataframe = dataframe.rename(index=str, columns={'Unnamed: 0': 'item_id'})
dataframe = dataframe.set_index('item_id')
x_data = dataframe

# make the problem supervised test_df is the train_df shifted -1
y_data = x_data.shift(-1)
y_data.fillna(0, inplace=True)


# drop S_day [:, :-1] or one hot rep and S_day [:, :-8] from training
x_data = y_data.iloc[:, :-8]

# drop last 8 columns referring to one hot rep from the target values
y_data = y_data.iloc[:, :-8]


# transform dataframes to np arrays
x_data = x_data.values
y_data = y_data.values


# model trains over 32 months and tests on 2 months
train_x, train_y = x_data[0:-61], y_data[0:-61]
test_x, test_y = x_data[-61:], y_data[-61:]


# calculates min max matrix for every column
def scaler(target_data):
    minmax = []
    n_columns = target_data.shape[1]
    n_rows = target_data.shape[0]
    print(n_rows, n_columns)
    for column in range(n_columns):
        value_min = min(target_data[:, column])
        value_max = max(target_data[:, column])
        minmax.append([value_min, value_max])
    return minmax


# scales data into [0, 1] according the matrix from scaler function
def scale_data(target_data, minmax):
    target_data = target_data.astype(float)
    n_columns = target_data.shape[1]
    n_rows = target_data.shape[0]
    for column in range(n_columns):
        for row in range(n_rows):
            if minmax[column][1] > 0:  # if max = 0 there is division error and column is for sure full of zero
                target_data[row, column] = (target_data[row, column] - minmax[column][0]) / (minmax[column][1] - minmax[column][0])
    return target_data


# inverse scaling using a min max matrix
def inverse_scaling(target_data, minmax_matrix):
    n_columns = target_data.shape[1]
    n_rows = target_data.shape[0]
    for column in range(n_columns):
        for row in range(n_rows):
            target_data[row, column] = (target_data[row, column] * (minmax_matrix[column][1] - minmax_matrix[column][0])) + minmax_matrix[column][0]
    return target_data


if SCALE_DATA == 1:
    scaler_train_x = scaler(train_x)
    scaler_train_y = scaler(train_y)
    scaler_test_x = scaler(test_x)
    scaler_test_y = scaler(test_y)

    train_x = scale_data(train_x, scaler_train_x)
    train_y = scale_data(train_y, scaler_train_y)
    test_x = scale_data(test_x, scaler_test_x)
    test_y = scale_data(test_y, scaler_test_y)


# shape data for LSTM model
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# design model
model = Sequential()
model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2]), stateful=False))
model.add(Dense(y_data.shape[1]))  # = 198 number of products
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


# create checkpoint callback to keep epoch with smallest val_loss
filepath = 'h5_files/simple_model_best.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# early stoping callback to stop training after a given number of epochs
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)


# fit model
history = model.fit(train_x, train_y, epochs=100,
                    validation_data=(test_x, test_y), verbose=2,
                    callbacks=[checkpoint, early_stopping_callback], shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# use the best forecast
model = load_model('h5_files/simple_model_best.h5')


# make a prediction
test_pred = model.predict(test_x)


if SCALE_DATA == 1:
    test_pred = inverse_scaling(test_pred, scaler_test_y)
    test_y = inverse_scaling(test_y, scaler_test_y)


# calculate mse of predicted and test
mse = mean_squared_error(test_y, test_pred)
print('Val MSE: %.3f' % mse)


# create plot with target everyday sales from all products vs predicted sales
test_y_list = []
test_pred_list = []
test_date_range = pandas.date_range(start='2015/09/01', end='2015/10/31')


for i in range(test_y.shape[0]):
    sum_per_day_y = 0
    sum_per_day_pred = 0
    for j in range(test_y.shape[1]):
        sum_per_day_y = sum_per_day_y + test_y[i, j]
        sum_per_day_pred = sum_per_day_pred + test_pred[i, j]
    test_y_list.append(sum_per_day_y)
    test_pred_list.append(sum_per_day_pred)

pyplot.plot(test_date_range, test_y_list, label='target')
pyplot.plot(test_date_range, test_pred_list, label='predicted')
pyplot.xlabel('Time', rotation=0)
pyplot.ylabel('Cumulative sales of all products')
pyplot.xticks(rotation=45)
pyplot.legend()
pyplot.show()

print("--- %s s ---" % (time.time() - start_time))
