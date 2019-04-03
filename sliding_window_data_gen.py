import numpy as np
import keras
from random import randint


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_of_ids, batch_size, in_dim, out_dim, weeks, days_per_shop, shuffle,):
        'Initialization'
        self.list_of_ids = list_of_ids
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weeks = weeks
        self.days_per_shop = days_per_shop
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_of_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Find list of IDs
        input_ids_list = []
        target_ids_list = []

        # loop through shops to find weeks
        for i in range(60):  # for every of the 60 shops
            for j in range(self.weeks):  # take a number of weeks
                pick = randint(0, self.days_per_shop-7)  # we need 7 days ahead of pick day so we avoid out of index error this way
                while (self.list_of_ids[i][pick]) in (input_ids_list or target_ids_list):  # maybe change for while (training_data[i][pick] for x in range (7) in (input_ids_list or target_ids_list):
                    pick = randint(0, self.days_per_shop-7)
                    #print('Duplicate pick again')
                #print(pick)
                input_ids_list.extend(self.list_of_ids[i][pick + x] for x in range(7))  # takes pick day and the next 6 days id
                target_ids_list.append(self.list_of_ids[i][pick + 7])  # takes target pick day (7 days after one week)

        # Generate data
        x, y = self.__data_generation(input_ids_list, target_ids_list)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_of_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, input_ids_list, target_ids_list):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([*self.in_dim])
        y = np.empty([self.out_dim])

        print(input_ids_list)
        print(target_ids_list)

        # Generate data prepei na ftiaksw batch to opoio tha periexei 32 x 7 grammes
        for i in range(7):
            # Store sample
            x[i, ] = np.load('data/data_rows/' + input_ids_list[i] + '.npy')

        #y[0, ] = np.load('data/data_rows/' + target_ids_list[0] + '.npy')

        input_ids_list = input_ids_list[7:]
        target_ids_list = target_ids_list[1:]

        return x, y
