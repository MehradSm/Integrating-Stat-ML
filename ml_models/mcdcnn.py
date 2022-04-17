# MCDCNN model

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from load_data import preprocessingData

##################################### Building Model #####################################
''' 
    Reference
    ----------
    @article{IsmailFawaz2018deep,
        Title                    = {Deep learning for time series classification: a review},
        Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
        journal                  = {Data Mining and Knowledge Discovery},
        Year                     = {2019},
        volume                   = {33},
        number                   = {4},
        pages                    = {917--963},
        }
'''

def MCDCNN(timesteps, data_dim):
    print('Creating model...')
    n_t = timesteps
    n_vars = data_dim
    padding = 'valid'

    input_layers = []
    conv2_layers = []

    for n_var in range(n_vars):
        input_layer = keras.layers.Input((n_t, 1))
        input_layers.append(input_layer)

        conv1_layer = keras.layers.Conv1D(filters=3, kernel_size=2, padding='same')(input_layer)
        conv1_layer = keras.layers.BatchNormalization()(conv1_layer)
        conv1_layer = keras.layers.Activation(activation='relu')(conv1_layer)

        conv2_layer = keras.layers.Conv1D(filters=3, kernel_size=2, padding='same')(conv1_layer)
        conv2_layer = keras.layers.BatchNormalization()(conv2_layer)
        conv2_layer = keras.layers.Activation(activation='relu')(conv2_layer)

        conv2_layer = keras.layers.Flatten()(conv2_layer)
        conv2_layers.append(conv2_layer)

    if n_vars == 1:
        # For uni-variate variables
        concat_layer = conv2_layers[0]
    else:
        concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

    fullyConnected = keras.layers.Dense(units=128, activation='relu')(concat_layer)
    dropOut = keras.layers.Dropout(0.5)(fullyConnected)

    output_layer = keras.layers.Dense(4, activation='softmax')(dropOut)

    model = keras.models.Model(inputs=input_layers, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def prepare_input(x):
    new_x = []
    n_t = x.shape[1]
    n_vars = x.shape[2]
    for i in range(n_vars):
        new_x.append(x[:,:,i:i+1])
    return  new_x

def indices(data, function):
    return [idx for (idx, val) in enumerate(data) if function(val)]
##################################### Main #####################################
X_train, y_train, X_test, y_test = preprocessingData()

y_test_copy = y_test

timesteps = X_train.shape[1]
data_dim = X_train.shape[2]

# One-Hot encoding for categorical vaiables
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

X_train =  prepare_input(X_train)
X_test =  prepare_input(X_test)

model = MCDCNN(timesteps, data_dim)

model.summary()

print('Fitting model...')
hist = model.fit(X_train, y_train, batch_size=128, epochs=1, validation_split=0.1, verbose=1)

score, acc = model.evaluate(X_test, y_test, batch_size=10)
label_ml = model.predict(X_test, batch_size=10)

print('Test score:', score)
print('Test accuracy:', acc)

# Test Accuracy by Class
idx_unf = indices(y_test_copy, lambda x:x==0)
idx_pos = indices(y_test_copy, lambda x:x==1)
idx_posspd = indices(y_test_copy, lambda x:x==2)
idx_posspddir = indices(y_test_copy, lambda x:x==3)

y_test_copy = keras.utils.to_categorical(y_test_copy)

X_test_unf = X_test[idx_unf,:]
X_test_unf = prepare_input(X_test_unf)
score_unf, acc_unf = model.evaluate(X_test_unf, y_test_copy[idx_unf,:], batch_size=10)

X_test_pos = X_test[idx_pos,:]
X_test_pos = prepare_input(X_test_pos)
score_pos, acc_pos = model.evaluate(X_test_pos, y_test_copy[idx_pos,:], batch_size=10)

X_test_posspd = X_test[idx_posspd,:]
X_test_posspd = prepare_input(X_test_posspd)
score_posspd, acc_posspd = model.evaluate(X_test_posspd, y_test_copy[idx_posspd,:], batch_size=10)

X_test_posspddir = X_test[idx_posspddir,:]
X_test_posspddir =  prepare_input(X_test_posspddir)
score_posspddir, acc_posspddir = model.evaluate(X_test_posspddir, y_test_copy[idx_posspddir,:], batch_size=10)

print('Test acc_unf:', acc_unf)
print('Test acc_pos:', acc_pos)
print('Test acc_posspd:', acc_posspd)
print('Test acc_posspddir:', acc_posspddir)

