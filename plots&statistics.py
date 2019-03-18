import pandas
import numpy as np
from matplotlib import pyplot as plt

DATA_TYPE = '.csv'
INPUT_PATH = 'data/processed/'
INPUT_DATA = 'PS4_SET_ALL_SHOPS'

dataframe = pandas.read_csv(INPUT_PATH + INPUT_DATA + DATA_TYPE)
dataframe = dataframe.rename(index=str, columns={'Unnamed: 0': 'item_id'})
dataframe = dataframe.set_index('item_id')

dataframe = dataframe.drop(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'S_Day'], axis=1)

print(dataframe)

item_values_count = dataframe.apply(pandas.value_counts)
print(item_values_count)

# number of zeros for every item
number_of_zeros = item_values_count.max(axis=0)
print(number_of_zeros)
print(max(number_of_zeros))
print(min(number_of_zeros))

plt.hist(item_values_count)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('yolo')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


