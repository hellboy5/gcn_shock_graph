from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
from keras.utils import plot_model
import keras.backend as K
import numpy as np

from keras_dgl.layers import MultiGraphCNN
from datagen.sg_graph_datagen import ShockGraphDataGenerator

train_dir='/home/naraym1/cifar_100/train_dir'
test_dir='/home/naraym1/cifar_100/test_dir'
label_file='/home/naraym1/cifar_100/labels.txt'

# Parameters
params = {'numb_nodes': 294,
          'numb_attrs': 19,
          'batch_size': 64,
          'n_classes': 100,
          'shuffle': True}

#Generators
training_generator=ShockGraphDataGenerator(train_dir,label_file,**params)
validation_generator=ShockGraphDataGenerator(test_dir,label_file,**params)

print(len(training_generator))
print(len(validation_generator))

# build model
feature_mat_input = Input(shape=(params['numb_nodes'], params['numb_attrs']))
adj_mat_input = Input(shape=(params['numb_nodes'],params['numb_nodes']))

output = MultiGraphCNN(100, 1, activation='relu')([feature_mat_input, adj_mat_input])
output = Dropout(0.2)(output)
output = MultiGraphCNN(100, 1, activation='relu')([output, adj_mat_input])
output = Dropout(0.2)(output)
output = MultiGraphCNN(100, 1, activation='relu')([output, adj_mat_input])
output = Dropout(0.2)(output)
output = MultiGraphCNN(100, 1, activation='relu')([output, adj_mat_input])
output = Dropout(0.2)(output)
output = MultiGraphCNN(100, 1, activation='relu')([output, adj_mat_input])
output = Dropout(0.2)(output)
# adding a node invariant layer to make sure output does not depends upon the node order in a graph.
output = Lambda(lambda x: K.mean(x, axis=1))(output)  
output = Dense(params['n_classes'])(output)
output = Activation('softmax')(output)

nb_epochs = 400

model = Model(inputs=[feature_mat_input, adj_mat_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

plot_model(model,to_file='sg_model.png',show_shapes=True)

model.fit_generator(generator=training_generator,
                    epochs=nb_epochs,
                    verbose=1,
                    validation_data=validation_generator,
                    shuffle=True)


