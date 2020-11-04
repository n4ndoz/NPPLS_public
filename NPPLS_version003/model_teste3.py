import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Bidirectional, Masking, Lambda, BatchNormalization
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Input, add
from layer_utils_3 import Alphabet
from tcn import TCN, tcn_full_summary

import os
from random import randint
import numpy as np
import tensorflow as tf


class CoreModels():
    @staticmethod
    def reslstm(inputs, rnn_width=180, rnn_depth=4, rnn_dropout=0.8, bn=True):
        x = inputs
        assert rnn_depth >= 2, "Parameter rnn_depth must be equal or greater than 2."
        for i in range(rnn_depth):
            #i < rnn_depth - 1
            x_rnn = Bidirectional(LSTM(rnn_width, recurrent_dropout=rnn_dropout,
                                       dropout=rnn_dropout, return_sequences=True))(x)
            if i > 0:
                x = add([x, x_rnn])
                x = BatchNormalization()(x) if bn else x
            else:
                x = x_rnn
        return x

    @staticmethod
    def bilstm(inputs, rnn_width=100, rnn_depth=2, rnn_dropout=0.2):
        x=inputs
        for cell_count in range(rnn_depth):
            x = Bidirectional(LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True))(x)
        return x

    @staticmethod
    def stacked_tcn(inputs, stacks=[64,32], nb_stacks=2, kernel_sizes=[2,2], dilations = [[1, 2, 4, 8, 16, 32],[1, 2, 4, 8, 16]]):
        '''
        Stacked Temporal Convolutional Network
        COnsists of stacks of TCNs, as implemented by Philipp Remy at
        https://github.com/philipperemy/keras-tcn. Parameters are:
        stacks: (list) are the numbers of filters
        nb_stacks: (int) the number of stacks of residual blocks on the TCN layer (lib stuff)
        kernel_sizes: (list) list of kernel sizes for the TCN stacks.
        
        '''
        x = inputs
        try:
            for stack,k_size,dils in zip(stacks,kernel_sizes, dilations):
                x = TCN(nb_filters=stack, kernel_size=2, nb_stacks=nb_stacks, dilations=[1, 2, 4, 8, 16],
                padding='causal', use_skip_connections=True, dropout_rate=0.2, return_sequences=True,
                activation='elu', kernel_initializer='he_normal', use_batch_norm=True)(x)
            return x
        except:
            print("Something wen't wrong while setting the TCN layer, review parameters.")
    

class AnglePredictor(Model):
    def __init__(self, core_model='TCN', ang_mode='alphabet',task='dih', **kwargs):
        '''
        AnglePredictor Class.
        This class is just a subclass of tf.keras.models.Model. In here I wrap up everything implemented
        on other classes (the alphabet stuff, the core models and other stuff in the future).
        '''
        super(AnglePredictor, self).__init__()
        self.layers_core = []
        if core_model == 'TCN':
            self.layers_core.append(TCN(nb_filters=128, kernel_size=1, nb_stacks=4, dilations=[1, 2, 4, 8, 16,32],
            padding='causal', use_skip_connections=True, dropout_rate=0.2, return_sequences=True,
            activation='relu', kernel_initializer='he_normal', use_batch_norm=True))
            self.layers_core.append(TCN(nb_filters=128, kernel_size=1, nb_stacks=4, dilations=[1, 2, 4, 8, 16,32],
            padding='causal', use_skip_connections=True, dropout_rate=0.2, return_sequences=True,
            activation='relu', kernel_initializer='he_normal', use_batch_norm=True))
            self.layers_core.append(TCN(nb_filters=64, kernel_size=2, nb_stacks=2, dilations=[1, 2, 4, 8, 16],
            padding='causal', use_skip_connections=True, dropout_rate=0.2, return_sequences=True,
            activation='elu', kernel_initializer='he_normal', use_batch_norm=True))
            self.layers_core.append(TCN(nb_filters=32, kernel_size=2, nb_stacks=2, dilations=[1, 2, 4, 8, 16],
            padding='causal', use_skip_connections=True, dropout_rate=0.2, return_sequences=True,
            activation='elu', kernel_initializer='he_normal', use_batch_norm=True))
        elif core_model== 'bilstm':
            for cell_count in range(4):
                self.layers_core.append(Bidirectional(LSTM(180, recurrent_dropout=0.3, dropout=0.2, return_sequences=True)))
                
        self.alphabet = Alphabet(n_alphas=40)
        self.masking = Masking(0.0)
    
    
    def call(self, inputs):
        x = self.masking(inputs)
        for layer in self.layers_core:
            x = layer(x)
        x = self.alphabet(x)
        return x
    
