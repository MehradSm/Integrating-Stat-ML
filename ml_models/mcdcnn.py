# CNN model

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import normalize

##################################### Loading Data #####################################
# Change this part for your machine and data
 
# 1. Spike trains for different categories, and for train and test
input_file_spike_unf_train = 'Data/spikeTrain_unf.csv'
input_file_spike_pos_train = 'Data/spikeTrain_pos.csv'
input_file_spike_speed_train = 'Data/spikeTrain_speed.csv'
input_file_spike_dir_train = 'Data/spikeTrain_dir.csv'

input_file_spike_unf_test = 'Data/spikeTest_unf.csv'
input_file_spike_pos_test = 'Data/spikeTest_pos.csv'
input_file_spike_speed_test = 'Data/spikeTest_speed.csv'
input_file_spike = 'Data/spike.csv'

# 2. Behavioral signals
input_file_pos = 'Data/pos.csv'
input_file_speed = 'Data/speed.csv'
input_file_direction = 'Data/direction.csv'

#3. Labels
input_file_label_train = 'Data/label_train.csv'
input_file_label = 'Data/label_posspddir.csv'
input_file_label_test = 'Data/label_test.csv'

def load_data():
    print('Loading data...')
    ##### Training Set
    df_spike_unf = pd.read_csv(input_file_spike_unf_train, header=None)
    df_spike_unf = df_spike_unf.transpose()
    df_spike_unf = df_spike_unf.fillna(0)

    df_spike_pos = pd.read_csv(input_file_spike_pos_train, header=None)
    df_spike_pos = df_spike_pos.transpose()
    df_spike_pos = df_spike_pos.fillna(0)

    df_spike_posspd = pd.read_csv(input_file_spike_speed_train, header=None)
    df_spike_posspd = df_spike_posspd.transpose()
    df_spike_posspd = df_spike_posspd.fillna(0)

    df_spike_posspddir = pd.read_csv(input_file_spike_dir_train, header=None)
    df_spike_posspddir = df_spike_posspddir.transpose()
    df_spike_posspddir = df_spike_posspddir.fillna(0)

    df_spike_train = pd.concat([df_spike_unf, df_spike_pos, df_spike_posspd, df_spike_posspddir], axis=0,
                               ignore_index=True)

    df_pos = pd.read_csv(input_file_pos, header=None)
    df_pos = df_pos.transpose()
    df_pos = df_pos.fillna(0)
    df_pos_train = df_pos[0:204]
    df_pos_train = pd.concat([df_pos_train, df_pos_train, df_pos_train, df_pos_train], axis=0, ignore_index=True)

    df_speed = pd.read_csv(input_file_speed, header=None)
    df_speed = df_speed.transpose()
    df_speed = df_speed.fillna(0)
    df_speed_train = df_speed[0:204]
    df_speed_train = pd.concat([df_speed_train, df_speed_train, df_speed_train, df_speed_train], axis=0,
                               ignore_index=True)

    df_direction = pd.read_csv(input_file_direction, header=None)
    df_direction = df_direction.transpose()
    df_direction = df_direction.fillna(0)
    df_direction[len(df_direction.columns) + 1] = 0
    df_direction_train = df_direction[0:204]
    df_direction_train = pd.concat([df_direction_train, df_direction_train, df_direction_train, df_direction_train],
                                   axis=0, ignore_index=True)

    df_label_train = pd.read_csv(input_file_label_train, header=None)

    # Shuffling traininig set
    np.random.seed(42)
    idx = np.random.permutation(df_spike_train.index)
    df_spike_train = df_spike_train.reindex(idx)
    df_pos_train = df_pos_train.reindex(idx)
    df_speed_train = df_speed_train.reindex(idx)
    df_direction_train = df_direction_train.reindex(idx)
    df_label_train = df_label_train.reindex(idx)

    # Normalize the feature space
    spike_train = np.array(df_spike_train.values)
    pos_train = np.array(df_pos_train.values)
    pos_train = normalize(pos_train)
    speed_train = np.array(df_speed_train.values)
    speed_train = normalize(speed_train)
    direction_train = np.array(df_direction_train.values)
    label_train = np.array(df_label_train.values)

    input_data_train = np.dstack((spike_train, pos_train, speed_train, direction_train))

    X_train = input_data_train
    y_train = label_train

    ##### Test Set
    df_spike_spl = pd.read_csv(input_file_spike, header=None)
    df_spike_spl = df_spike_spl.transpose()
    df_spike_spl = df_spike_spl.fillna(0)
    df_spike_spl = df_spike_spl.replace(2, 1)
    df_spl_test = df_spike_spl[204:]

    df_spike_unf_test = pd.read_csv(input_file_spike_unf_test, header=None)
    df_spike_unf_test = df_spike_unf_test.transpose()
    df_spike_unf_test = df_spike_unf_test.fillna(0)

    df_spike_pos_test = pd.read_csv(input_file_spike_pos_test, header=None)
    df_spike_pos_test = df_spike_pos_test.transpose()
    df_spike_pos_test = df_spike_pos_test.fillna(0)

    df_spike_speed_test = pd.read_csv(input_file_spike_speed_test, header=None)
    df_spike_speed_test = df_spike_speed_test.transpose()
    df_spike_speed_test = df_spike_speed_test.fillna(0)

    df_spike_test = pd.concat([df_spike_unf_test, df_spike_pos_test, df_spike_speed_test, df_spl_test], axis=0,
                              ignore_index=True)

    df_pos_test = df_pos[204:]
    df_pos_test = pd.concat([df_pos_test, df_pos_test, df_pos_test, df_pos_test], axis=0, ignore_index=True)

    df_speed_test = df_speed[204:]
    df_speed_test = pd.concat([df_speed_test, df_speed_test, df_speed_test, df_speed_test], axis=0, ignore_index=True)

    df_direction_test = df_direction[204:]
    df_direction_test = pd.concat([df_direction_test, df_direction_test, df_direction_test, df_direction_test], axis=0,
                                  ignore_index=True)

    df_label_test = pd.read_csv(input_file_label_test, header=None)
    df_label = pd.read_csv(input_file_label, header=None)
    df_label_test = pd.concat([df_label_test, df_label[204:]], axis=0, ignore_index=True)

    # Normalize the feature space
    spike_test = np.array(df_spike_test.values)
    pos_test = np.array(df_pos_test.values)
    pos_test = normalize(pos_test)
    speed_test = np.array(df_speed_test.values)
    speed_test = normalize(speed_test)
    direction_test = np.array(df_direction_test.values)
    label_test = np.array(df_label_test.values)

    input_data_test = np.dstack((spike_test, pos_test, speed_test, direction_test))

    X_test = input_data_test
    y_test = label_test

    print('Loading Finished')
    return X_train, y_train, X_test, y_test

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
X_train, y_train, X_test, y_test = load_data()

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
hist = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.1, verbose=1)

score, acc = model.evaluate(X_test, y_test, batch_size=10)
label_ml = model.predict(X_test, batch_size=10)

print('Test score:', score)
print('Test accuracy:', acc)

# Test Accuracy by Class
idx_unf = indices(y_test, lambda x:x==0)
idx_pos = indices(y_test, lambda x:x==1)
idx_posspd = indices(y_test, lambda x:x==2)
idx_posspddir = indices(y_test, lambda x:x==3)

X_test_unf = X_test[idx_unf,:]
X_test_unf = prepare_input(X_test_unf)
score_unf, acc_unf = model.evaluate(X_test_unf, y_test[idx_unf,:], batch_size=10)

X_test_pos = X_test[idx_pos,:]
X_test_pos = prepare_input(X_test_pos)
score_pos, acc_pos = model.evaluate(X_test_pos, y_test[idx_pos,:], batch_size=10)

X_test_posspd = X_test[idx_posspd,:]
X_test_posspd = prepare_input(X_test_posspd)
score_posspd, acc_posspd = model.evaluate(X_test_posspd, y_test[idx_posspd,:], batch_size=10)

X_test_posspddir = X_test[idx_posspddir,:]
X_test_posspddir =  prepare_input(X_test_posspddir)
score_posspddir, acc_posspddir = model.evaluate(X_test_posspddir, y_test[idx_posspddir,:], batch_size=10)

print('Test acc_unf:', acc_unf)
print('Test acc_pos:', acc_pos)
print('Test acc_posspd:', acc_posspd)
print('Test acc_posspddir:', acc_posspddir)


