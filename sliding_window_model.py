import pandas
from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from sklearn.preprocessing import MinMaxScaler

DATA_TYPE = '.csv'
INPUT_PATH = 'data/processed/'
INPUT_DATA = 'PS4_SET_ALL_SHOPS'

dataframe = pandas.read_csv(INPUT_PATH + INPUT_DATA + DATA_TYPE)
dataframe = dataframe.rename(index=str, columns={'Unnamed: 0': 'item_id'})
dataframe = dataframe.set_index('item_id')
x_data = dataframe


# make the problem supervised test_df is the train_df shifted -1
y_data = x_data.shift(-1)
y_data.fillna(0, inplace=True)

# take the 2 last months for test
test_date_range = pandas.date_range(start='2015/09/01', end='2015/10/31')
test_n_days = test_date_range.nunique()
n_shops = 60

# change dimensions in order to prepare data
x_data = x_data.T
y_data = y_data.T

# initialize empty test_x and test_y data frames
test_x = pandas.DataFrame(index=range(0))
test_y = pandas.DataFrame(index=range(0))

for i in range(n_shops):
    for j in range(test_n_days):
        date = str(pandas.to_datetime(test_date_range[j]).date())

        df1 = x_data[date]
        df1 = df1.iloc[:, i:i+1]
        test_x = pandas.concat([test_x, df1], axis=1)  # ,sort=True

        df2 = y_data[date]
        df2 = df2.iloc[:, i:i+1]
        test_y = pandas.concat([test_y, df2], axis=1)  # ,sort=True


# drop the test part from the x_data and y_data data frames
for j in range(test_n_days):
    x_data = x_data.drop([str(pandas.to_datetime(test_date_range[j]).date())], axis=1)
    y_data = y_data.drop([str(pandas.to_datetime(test_date_range[j]).date())], axis=1)

# x_data and y_data contain whole data minus test
train_x = x_data
train_y = y_data

# return data to original dimensions
train_x = train_x.T
test_x = test_x.T
train_y = train_y.T
test_y = test_y.T

print(train_x)
print(train_y)
print(test_x)
print(test_y)


# scale data in [-1, 1]
#scaler = MinMaxScaler(feature_range=(-1, 1))
#train_x = scaler.fit_transform(train_x)
#test_x = scaler.fit_transform(test_x)
#train_y = scaler.fit_transform(train_y)
#test_y = scaler.fit_transform(test_y)

