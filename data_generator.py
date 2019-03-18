import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_ids, target_ids, batch_size, shuffle, dim):
        'Initialization'
        self.batch_size = batch_size
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.shuffle = shuffle
        self.dim = dim
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.input_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        input_ids_temp = [self.input_ids[k] for k in indexes]
        target_ids_temp = [self.target_ids[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(input_ids_temp, target_ids_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.input_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, input_ids_temp, target_ids_temp):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, *self.dim])
        y = np.empty([self.batch_size, *self.dim])

        # Generate data
        for i in range(len(input_ids_temp)):
            # Store sample
            x[i, ] = np.load('data/x_data/' + input_ids_temp[i] + '.npy')
            y[i, ] = np.load('data/y_data/' + target_ids_temp[i] + '.npy')

        return x, y
