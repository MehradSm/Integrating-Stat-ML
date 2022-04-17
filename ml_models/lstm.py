# LSTM model

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from load_data import preprocessingData
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

##################################### Building Model #####################################
def LSTMM(timesteps, data_dim):
    print('Creating model...')
    model = Sequential()
    model.add(LSTM(6, activation='sigmoid',
                   input_shape=(timesteps, data_dim), return_sequences=True))
    model.add(LSTM(4, activation='sigmoid'))
    model.add(Dense(4, activation='softmax'))

    print('Compiling...')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

##################################### Main #####################################
X_train, y_train, X_test, y_test = preprocessingData()

timesteps = X_train.shape[1]
data_dim = X_train.shape[2]

# One-Hot encoding for categorical vaiables
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = LSTMM(timesteps, data_dim)

model.summary()

print('Fitting model...')
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.1, verbose=1)

score, acc = model.evaluate(X_test, y_test, batch_size=10)

print('Test score:', score)
print('Test accuracy:', acc)
