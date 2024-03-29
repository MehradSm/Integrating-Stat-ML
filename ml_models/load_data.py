import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize


# Change this part for your machine and data

# 1. Spike trains for different categories, and for train and test
input_file_spike_unf_train = 'Data/spikeTrain_unf_multi.csv'
input_file_spike_pos_train = 'Data/spikeTrain_pos_multi.csv'
input_file_spike_speed_train = 'Data/spikeTrain_speed_multi.csv'
input_file_spike_dir_train = 'Data/spikeTrain_dir_multi.csv'

input_file_spike_unf_test = 'Data/spikeTest_unf_multi.csv'
input_file_spike_pos_test = 'Data/spikeTest_pos_multi.csv'
input_file_spike_speed_test = 'Data/spikeTest_speed_multi.csv'
input_file_spike = 'Data/spike.csv'

# 2. Behavioral signals
input_file_pos = 'Data/pos.csv'
input_file_speed = 'Data/speed.csv'
input_file_direction = 'Data/direction.csv'

#3. Labels
input_file_label_train = 'Data/label_train_unfposspeeddir.csv'
input_file_label = 'Data/label_posspddir.csv'
input_file_label_test = 'Data/label_unfposspeed.csv'

def preprocessingData():
    print('Loading and preprocessing Data ...')
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

    # Normalizing the feature space
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

    # Normalizing the feature space
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
