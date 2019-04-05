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

print(training_data)
print(validation_data)

'''
def random_samples(n_shops, n_weeks, n_days, data):
    input_ids_list = []
    target_ids_list = []

    # loop through shops to find weeks
    for l in range(n_shops):
        print(l)
        for m in range(n_weeks):  # take a number of weeks
            # last day is the day 972 but we need 7 days ahead of pick day so we avoid out of index error this way
            pick = randint(0, n_days-7)
            # maybe change for while (training_data[i][pick] for x in range (7) in (input_ids_list or target_ids_list):
            while (data[i][pick]) in (
                    input_ids_list or target_ids_list):
                pick = randint(0, n_days-7)
                #print('Duplicate pick again')
            #print(pick)
            input_ids_list.extend(data[i][pick + x] for x in range(7))  # takes pick day and the next 6 days id
            target_ids_list.append(data[i][pick + 7])  # takes target pick day (7 days after one week)

    return input_ids_list, target_ids_list


training_input_ids_list, training_target_ids_list = random_samples(60, 32, 972, training_data)
validation_input_ids_list, validation_target_ids_list = random_samples(60, 4, 60, validation_data)

print(training_input_ids_list)
print(len(training_input_ids_list))
print(training_target_ids_list)
print(len(training_target_ids_list))
# kai dinw 2 listes gia train 2 gia val kai trexw etsi
'''

# Parameters
params_train = {'batch_size': 32,
                'in_dim': (7, 215),
                'out_dim': 215,
                'days_per_shop': 972,
                'shuffle': False}


params_val = {'batch_size': 4,
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
