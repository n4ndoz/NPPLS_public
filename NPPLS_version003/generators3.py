from os import path
import tensorflow.keras as keras
import numpy as np

class AngleDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, dim=(256,46), n_angles=1, data_dir='latest_train_ind', ang_type='theta', shuffle=True):
        'Initialization'
        self.ang_type = ang_type
        self.data_dir=data_dir
        self.dim = dim
        self.n_angles=n_angles # sets if only phi and psi (2) or includes omega (3)
        self.batch_size = batch_size
        self.data=data # a dictionary containing x_path and y_path
        self.list_IDs = list(self.data.keys())
        self.shuffle = shuffle
        self.on_epoch_end()
        # ok, this is inefficient, but will work for now
        self.angle = {'theta':1,'phi':2}[self.ang_type]

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
        #X = np.empty((self.batch_size, *self.dim))
        # preciso arrumar isso aqui pra colocar caso seja predição
		# de um angulo só. Coloco só como output shape ou entao pelo tipo
        X = np.empty((self.batch_size, *self.dim))
        #y = np.empty((self.batch_size,self.dim[0],self.n_angles))
        y = np.empty((self.batch_size,self.dim[0],self.n_angles))
        # Generate data
        # This is just the inde of the axis where the desired angle is located
        
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(path.join(self.data_dir,ID))
            # Store class
			# Just change this after you change at data_prep
			
			# Depois substituir isso por:
			# types = {'theta':1,'phi': 2}
			# y[i,]... [:,types[self.ang_type]
			# ou substituir por gather. Acho que vai ser mais da hora!
            
            
            # Ok! This approach is not efficient and needs to load the entire data at once.
            # I need look for a way to insert multiple outputs on a keras generator
            # Better! I need to adequate to TF data. But this will be a whole lotta different procedure
            # As we do not load the entire dataset at once, and the size of each entry is small, it is a usable approach.
            # But not a good approach. It is better to split the files. But not for now
            if self.ang_type == 'dih':
                y[i,] = np.load(path.join(self.data_dir, self.data.get(ID)))
            elif self.ang_type in ('theta','psi'):
                y[i,] = np.load(path.join(self.data_dir, self.data.get(ID)))[:,angle].reshape((-1,1)) # so as to get andles
            '''
            if self.ang_type=='theta':
                y[i,] = np.load(path.join(self.data_dir, self.data.get(ID)))[:,1].reshape((-1,1)) # so as to get andles
                                   
            elif self.ang_type=='phi':
                y[i,] = np.load(path.join(self.data_dir, self.data.get(ID)))[:,2].reshape((-1,1))
            '''
            
        return X, y
