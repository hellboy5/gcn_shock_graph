from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
from keras.utils import plot_model
import keras.backend as K
import numpy as np
import sys
import h5py

sys.path.append('/Users/naraym1/work/gcn_shock_graph')

from datagen.sg_graph_datagen import ShockGraphDataGenerator

train_dir='/Users/naraym1/work/cifar_100/unit_train_dir'
label_file='/Users/naraym1/work/cifar_100/scripts/labels.txt'

# Parameters
params = {'numb_nodes': 294,
          'numb_attrs': 19,
          'numb_filters':2,
          'batch_size': 100,
          'n_classes': 100,
          'shuffle': False}

#Generators
training_generator=ShockGraphDataGenerator(train_dir,label_file,**params)


batch,labels=training_generator[0]

hf = h5py.File('test.h5','w')


hf.create_dataset('batch',data=batch[1])
hf.create_dataset('labels',data=labels)

hf.close()
