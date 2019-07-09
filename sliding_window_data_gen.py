import numpy as np
import keras
from random import randint


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_of_ids, batch_size, in_dim, out_dim, days_per_shop, data_type, shuffle):
        'Initialization'
        self.list_of_ids = list_of_ids
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.days_per_shop = days_per_shop
        self.data_type = data_type
        self.shuffle = shuffle
        self.input_ids_list, self.target_ids_list = self.random_day()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # input_ids_list contains 60xbatch_sizex7 data
        return int(np.floor(len(self.input_ids_list) / (self.batch_size*7)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        input_indexes = self.input_indexes[index * self.batch_size * 7:(index + 1) * self.batch_size * 7]
        target_indexes = self.target_indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print(input_indexes, target_indexes)
        # print(index)

        # print(self.input_ids_list)
        # print(self.target_ids_list)
        # Find list of ID
        input_ids_list_temp = [self.input_ids_list[k] for k in input_indexes]
        target_ids_list_temp = [self.target_ids_list[k] for k in target_indexes]
        # print(input_ids_list_temp)
        # print(target_ids_list_temp)

        # Generate data
        x, y = self.__data_generation(input_ids_list_temp, target_ids_list_temp)
        # save data to plot it later
        if self.data_type == 'test':
            np.save('data/data_rows_target/' + 'shop_' + str(index) + '_target' + '.npy', y)

        return x, y

    def random_day(self):
        # list containing input and target data
        input_ids_list = []
        target_ids_list = []

        # loop through shops to find weeks
        for i in range(60):  # for every of the 60 shops
            for j in range(self.batch_size):  # take a number of weeks
                # we need 7 days ahead of pick day so we avoid out of index error this way
                pick = randint(0, self.days_per_shop - 7)
                # maybe change for while (training_data[i][pick] for x in range (7)
                # in (input_ids_list or target_ids_list): ws exei twra einai epikaluptomenes vdomades
                while (self.list_of_ids[i][pick]) in (input_ids_list or target_ids_list):
                    pick = randint(0, self.days_per_shop - 7)
                    # print('Duplicate pick again')
                # print(pick)
                input_ids_list.extend(self.list_of_ids[i][pick + x] for x in range(7))  # takes pick day and the next 6 days id
                target_ids_list.append(self.list_of_ids[i][pick + 7])  # takes target pick day (7 days after one week)
        return input_ids_list, target_ids_list

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.input_indexes = np.arange(len(self.input_ids_list))
        self.target_indexes = np.arange(len(self.target_ids_list))
        if self.shuffle == True:
            np.random.shuffle(self.input_indexes)
            np.random.shuffle(self.target_indexes)

    def __data_generation(self, input_ids_list_temp, target_ids_list_temp):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, *self.in_dim])
        y = np.empty([self.batch_size, self.out_dim])

        for j in range(self.batch_size):
            for i in range(7):
                # Store sample
                x[j, i, ] = np.load('data/data_rows/' + input_ids_list_temp[i] + '.npy')
            y[j] = np.load('data/data_rows/' + target_ids_list_temp[j] + '.npy')

        return x, y
