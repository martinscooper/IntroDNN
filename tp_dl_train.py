'''
El dataset se encuentra en dataset.pkl
train_all entrena 3 arquitecturas para 3 usuarios, guarda los modelos entrenados
y los datos de testeo en un diccionario a modo de cache

test_all y test se encargan de testear todos los modelos y mostrar el mse

'''

from pandas import concat
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, LSTM, Dropout, Dense, Flatten, BatchNormalization, Activation, Input
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, Model
import numpy as np
import pickle
from tcn import compiled_tcn


def get_architecture(n):
    model = Sequential()
    input_shape = (time_lags[n], number_of_features)
    if n == 1:
        model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(.4))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dropout(.2))
        model.add(Dense(8))
        model.add(Activation('relu'))
        model.add(Dense(1, activation='linear'))

    elif n == 2:
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(.3))
        model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(.6))
        model.add(Dense(1, activation='linear'))

    elif n == 3:
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(.8))
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(.8))
        model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(.8))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(.4))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dropout(.2))
        model.add(Dense(8))
        model.add(Activation('relu'))
        model.add(Dense(1, activation='linear'))

    elif n == 4:
        model = compiled_tcn(return_sequences=False,
                             num_feat=number_of_features,
                             num_classes=0,
                             nb_filters=64,
                             kernel_size=4,
                             dilations=[2 ** i for i in range(2, 5)],
                             nb_stacks=2,
                             max_len=time_lags[n],
                             activation='norm_relu',
                             use_skip_connections=True,
                             regression=True,
                             dropout_rate=0)
    elif n == 5:
        model = Sequential()
        model.add(Conv1D(64, 6, activation='relu', padding='causal', input_shape=(time_lags[n], number_of_features)))
        model.add(Conv1D(128, 12, activation='relu', padding='causal'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer='adam',
                      )
    elif n == 6:
        model.add(Dense(64, input_dim=29, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mse',
                  optimizer='adam',
                  )
    print(model.summary())
    return model


def get_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')


def shift_hours(df, n, columns=None):
    dfcopy = df.copy().sort_index()
    if columns is None:
        columns = df.columns
    for ind, row in dfcopy.iterrows():
        try:
            dfcopy.loc[(ind[0], ind[1]), columns] = dfcopy.loc[(ind[0], ind[1] + pd.DateOffset(hours=n)), columns]
        except KeyError:
            dfcopy.loc[(ind[0], ind[1]), columns] = np.nan
    # print(dfcopy.isna().sum())
    dfcopy.dropna(inplace=True)
    return dfcopy


def series_to_supervised(df2, dropnan=True, number_of_lags=None):
    lags = range(number_of_lags, 0, -1)
    columns = df2.columns
    n_vars = df2.shape[1]
    cols, names = list(), list()
    print('Generating time-lags...')
    # input sequence (t-n, ... t-1)
    for i in lags:
        cols.append(shift_hours(df2, i, df.columns))
        names += [('{0}(t-{1})'.format(columns[j], i)) for j in range(n_vars)]
    cols.append(df2)
    names += [('{0}(t)'.format(columns[j])) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def train_all(print_images=False):
    numeric_cols = ['stationaryLevel', 'walkingLevel', 'runningLevel',
                    'numberOfConversations', 'wifiChanges',
                    'silenceLevel', 'voiceLevel', 'noiseLevel',
                    'hourSine', 'hourCosine',
                    'remainingminutes', 'pastminutes',
                    'distanceTraveled', 'locationVariance']

    ss = StandardScaler()
    for i in users:
        print('Comienzan los entrenamientos con el usuario {0}'.format(i))
        userdata = get_user_data(df, i)
        userdata.loc[:, numeric_cols] = ss.fit_transform(userdata[numeric_cols])
        cache[i] = {}
        lags = -1
        for j in range(1, number_of_architectures + 1):
            print('El entrenamiendo del usuario {0} con la aquitectura {1} está por comenzar'.format(i, j))
            if lags != time_lags[j]:
                data = series_to_supervised(userdata, number_of_lags=time_lags[j])
            lags = time_lags[j]
            model = get_architecture(j)

            x = data.iloc[:, 0:time_lags[j] * number_of_features]
            y = data.iloc[:, -1]

            x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.67)
            x_train, y_train, x_test, y_test = x_train.values.astype("float32"), y_train.values.astype("float32"),\
                                               x_test.values.astype("float32"), y_test.values.astype("float32")

            if time_lags[j]>1:
                x_train = x_train.reshape(x_train.shape[0], time_lags[j], number_of_features)
                x_test = x_test.reshape(x_test.shape[0], time_lags[j], number_of_features)
            print('{0} casos de entrenamiento. **** {1} casos para testeo.'.format(x_train.shape[0], x_test.shape[0]))
            history = model.fit(x_train, y_train, epochs=64, batch_size=batch_size[j], validation_data=(x_test, y_test),
                                verbose=0)

            cache[i][j] = {'x_test': x_test, 'y_test': y_test, 'x_train': x_train, 'y_train': y_train,
                           'model': model, 'history': history}

            if print_images:
                show_images(i, j)

            print('El entrenamiendo del usuario {0} con la aquitectura {1} ha finalizado'.format(i, j))


def test_error(user, architecture):
    info = cache[user][architecture]
    x_test, y_test, model = info['x_test'], info['y_test'], info['model']
    y_pred = model.predict(x_test)
    return round(mean_squared_error(y_test, y_pred), 3)


def train_error(user, architecture):
    info = cache[user][architecture]
    x_train, y_train, model = info['x_train'], info['y_train'], info['model']
    y_pred = model.predict(x_train)
    return round(mean_squared_error(y_train, y_pred), 3)


def test_all():
    print('')
    print('*' * 16)
    for i in users:
        for j in range(number_of_architectures):
            mse = test_error(i, j + 1)
            print('El mse para el usuario {0} utilizando la arquitectura {1} fue: {2}'.format(i, j + 1, mse))

    print('*' * 16)
    print('')


def show_train_prediction(user, architecture):
    info = cache[user][architecture]
    plt.close()
    plt.figure(figsize=(15, 4))
    plt.title('Train data of user {0} with architecture {1}'.format(user, architecture))
    y_pred = info['model'].predict(info['x_train'])
    plt.plot(info['y_train'], label='Train')
    plt.plot(y_pred, label='Predicted')
    plt.axhline(y=1.5, color='r', linestyle=':', )
    plt.legend(loc='upper right')
    plt.show()


def show_test_prediction(user, architecture):
    info = cache[user][architecture]
    plt.close()
    plt.figure(figsize=(15, 4))
    plt.title('Test data of user {0} with architecture {1}'.format(user, architecture))
    y_pred = info['model'].predict(info['x_test'])
    plt.plot(info['y_test'], label='Test')
    plt.plot(y_pred, label='Predicted')
    plt.axhline(y=1.5, color='r', linestyle=':', )
    plt.legend(loc='upper right')
    plt.show()


def show_history_loss(user, architecture):
    history = cache[user][architecture]['history']
    plt.close()
    plt.title('Train loss vs. Test loss of user {0} with architecture {1}'.format(user, architecture))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def show_images(user, architecture):
    # info = pickle.load(open('testing_info.pkl', 'rb'))
    show_train_prediction(user, architecture)
    show_test_prediction(user, architecture)
    show_history_loss(user, architecture)


df = pd.read_pickle('dataset.pkl')

number_of_architectures = 6
users = [50, 31, 4]
batch_size = {1: 32, 2: 64, 3: 64, 4: 64, 5: 64, 6: 128}
time_lags = {1: 6, 2: 12, 3: 12, 4: 48, 5: 48, 6 : 1}
number_of_features = df.shape[1]

cache = {}
train_all(print_images=False)

pickle.dump(cache, open('testing_info.pkl', 'wb'))

