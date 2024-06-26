import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import glob
import random


class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_examples, batch_size=128, dim=(1, ),
                 n_classes=2, shuffle=True):
        'Initialization'
        print("Constructor called!!!")
        self.dim = dim
        self.batch_size = batch_size
        self.list_examples = list_examples
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        #print("The self.list_examples is {}".format(self.list_examples))
        return int(np.floor(len(self.list_examples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_examples[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
      self.indexes = np.arange(len(self.list_examples))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # # Initialization

        # X = np.empty([self.batch_size, 802, 80], dtype=np.float32)

        #new_model 
        X = np.empty([self.batch_size, 802, 100], dtype=np.float32)
        y = np.empty([self.batch_size, 802, 2], dtype=np.int16)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            X[i,:, :] = np.load(ID[0])

            # Store class          
            y[i, :, :] = np.load(ID[1])

        return X, y