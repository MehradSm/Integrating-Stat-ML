# MLP model

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from load_data import preprocessingData
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

############################## Building Model ##################################
def MLP(n_input):
    print ('Creating model...')
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=n_input))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                      metrics=['accuracy'])
    return model

##################################### Main #####################################
X_train, y_train, X_test, y_test = preprocessingData()

n_input = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0],n_input))

# One-Hot encoding for categorical vaiables
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = MLP(n_input)

model.summary()

print('Fitting model...')
model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.1, verbose=1)

score, acc = model.evaluate(X_test, y_test, batch_size=10)

print('Test score:', score)
print('Test accuracy:', acc)
