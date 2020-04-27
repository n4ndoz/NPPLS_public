import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Bidirectional, Masking, Lambda, LSTM, Dense, RepeatVector, TimeDistributed
import os
from random import randint
import numpy as np


class AnglePredictor:
    def __init__(self, ang_mode=[],net_type=''):
        # default the parameters so we could setup everything properly fomr now o        
        self.modes = {'tanh': ['tanh',Lambda(lambda x: x*np.pi)],
                      'sigmoid':['sigmoid',Lambda(lambda x: (x-0.5)*2*np.pi)],
                      'cluster_k10':['softmax','no_lambda',10]} # for now cluster_k10 is out
         # stores the model object that will be passed
        
    def BiLSTM(self, ang_mode=[], layers=[180,180,180,180], input_shape=[]):
        # ang_mode is a list containing a Lambda function and an activation
        # function name
        # layers is a list containing the cell count for each BiLSTM layer
        # it is defaulted to 4 180 cell layers
        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=input_shape))
        
        for cell_count in layers:
            model.add(Bidirectional(LSTM(cell_count, return_sequences=True)))

        # Gotta revamp all this ang_mode mechanics later
        if ang_mode[1] != 'no_lambda':
            model.add(TimeDistributed(Dense(2,activation=ang_mode[0])))
            model.add(Lambda(ang_mode[1]))
        else: # in this case, model is cluster_kN and we should change
            model.add(TimeDistributed(Dense(ang_mode[2],activation=ang_mode[0])))
        return model 

    def build_model(self,ang_mode, model_type,input_shape):
        models={'bilstm':self.BiLSTM}
        model = models[model_type](self.modes[ang_mode],input_shape=input_shape)
        return model

    # Define a print_avialable_models functions
    def print_available_modes(self):
        pass
