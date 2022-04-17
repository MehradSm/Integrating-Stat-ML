# CNN model

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from load_data import preprocessingData
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D

################################ Building Model ################################
def CNN(timesteps, data_dim):
    print('Creating model...')
    model = Sequential()
    model.add(Conv1D(10, 3, activation='relu', input_shape=(timesteps, data_dim)))
    model.add(Conv1D(7, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(4, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(4, activation='softmax'))

    print('Compiling...')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

#################################### Main #####################################
X_train, y_train, X_test, y_test = preprocessingData()

timesteps = X_train.shape[1]
data_dim = X_train.shape[2]

# One-Hot encoding for categorical vaiables
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


model = CNN(timesteps, data_dim)

model.summary()

print('Fitting model...')
model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.1, verbose=1)

score, acc = model.evaluate(X_test, y_test, batch_size=10)

print('Test score:', score)
print('Test accuracy:', acc)
