# training pipeline

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from generators import AngleDataGenerator
from Utils import Utils as utils
import os
import model2

'''
############### Data Loading Block ##################
'''

all_files = os.listdir('latest_train_ind')
sequence_dih_map = { 'x_{:04d}.npy'.format(i+1):'y_{:04d}.npy'.format(i+1) for
                     i in np.arange(len(all_files)//4)}

'''
############## Model preparation/declaration ##############
'''

mod = model2.AnglePredictor()
model = mod.build_model('tanh','bilstm',[500,46])
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])


'''
############# Training Step Preparations ################
'''
# This split is a little bit off
# This is just for testing
train_ids = dict(list(sequence_dih_map.items())[0:200])
valid_ids = dict(list(sequence_dih_map.items())[300:400])
train_gen = AngleDataGenerator(train_ids)
valid_gen = AngleDataGenerator(valid_ids)

'''
############ Training Routine ##################
'''

history = model.fit(x=train_gen,
          epochs=1,
          validation_data=valid_gen,
          use_multiprocessing=True,
         workers = 6)

# Compute some statistics on the model
train_losses = history.history['loss']
train_acc = history.history['accuracy']
val_losses = history.history['val_loss']
val_acc = history.history['val_accuracy']

# Save statistics
utils.save_batch({'tl':train_losses, 'ta': train_acc, 'vl': val_losses, 'va':val_acc},'',1)
