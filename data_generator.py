import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_ids, batch_size=32, dim=(32, 32, 32), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim)
        # Initialization
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # Store sample
            x[i, ] = np.load('data/' + ID + '.npy')
            y[i, ] = np.load('data/' + ID + '.npy')

        return x, y
