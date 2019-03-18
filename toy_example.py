import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense

train_x = np.random.rand(60, 973, 215)  # 1000 7 215 for sliding window
train_y = np.random.rand(60, 973, 215)  # 1000 215 for sliding window
#test_x = np.random.rand(10, 973, 215)  # 50 7 215 for sliding window
test_y = np.random.rand(10, 973, 215)  # 50 215 for sliding window

test_x = np.random.rand(610, 215)  # 50 7 215 for sliding window
print(test_x.shape)
# pad((top, bottom), (left, right)) (9730-610,0), (0,0)
test_x = np.pad(test_x, ((9730-610, 0), (0, 0)), 'constant', constant_values=0)
print(test_x.shape)
test_x = test_x.reshape(10, 973, 215)
print(test_x.shape)
print(test_x)


# design model
model = Sequential()
model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))  # return_sequence=False for sliding window
model.add(Dense(215))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# fit model
history = model.fit(train_x, train_y, epochs=10,
                    validation_data=(test_x, test_y), verbose=2, shuffle=True)
