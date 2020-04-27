import subprocess
import sys
import os
from os import listdir
import numpy as np

from Utils import Utils as utils

# This part comprehends the installation and importing of required packages
# (ok, there are some that do not belong here, but for now I need them installed
# for testing, will take it out ASAP) - 23/04/2020
# Function for installation of libraries via pip
def install_pkg(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

pkgs = ['scikit-learn==0.20.0','tqdm','scipy','seaborn','regex','biopython']
'''
for package in pkgs:
    try:
        import package
    except ImportError:
        print('Trying to install Package: {}'.format(package))
        install_pkg(package)
'''
import tensorflow as tf
from tqdm import tqdm
import numpy as np

import datagen_v002 as data_prep

# data_files dict holds 'mode':['data_file',output_folder]
data_files = {'train':['/home/fernando/storage/model1/Model1_pipeline/casp7/training_50','train_70'],
              'valid':['/home/fernando/storage/model1/Model1_pipeline/casp7/validation','valid_test'],
               'test':['/home/fernando/storage/model1/Model1_pipeline/casp7/testing','testing_data/']}

for k in data_files.keys():
    print('Preparing file * {} * | storing at * {} *'.format(data_files[k][0].partition('//')[-1],data_files[k][1]))
    data_prep = data_prep.Data_Prep_Pipeline(input_file=data_files[k][0])
    data_prep.make_batch(1,folder_name=data_files[k][1])

print("Finished prep. . . ")
