from os import path
import tensorflow.keras as keras
import numpy as np

class AngleDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, dim=(500,46), n_angles=2, data_dir='latest_train_ind', shuffle=True):
        'Initialization'
        self.data_dir=data_dir
        self.dim = dim
        self.n_angles=n_angles # sets if only phi and psi (2) or includes omega (3)
        self.batch_size = batch_size
        self.data=data # a dictionary containing x_path and y_path
        self.list_IDs = list(self.data.keys())
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        'gets the number of batches of batch_size length'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size,self.dim[0],self.n_angles), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(path.join(self.data_dir,ID))

            # Store class
            y[i,] = np.load(path.join(self.data_dir, self.data.get(ID)))

        return X, y
