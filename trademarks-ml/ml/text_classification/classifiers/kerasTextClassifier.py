import numpy as np

from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import LSTM,Bidirectional,TimeDistributed,InputLayer

def create_sequential_model(optimizer='adagrad',kernel_initializer='glorot_uniform',dropout=0.2):

    model = Sequential()
    model.add(Dense(64,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid',kernel_initializer=kernel_initializer))
    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def create_bi_lstm_model(optimizer='adagrad',kernel_initializer='glorot_uniform',dropout=0.2, mode='concat'):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences = True), input_shape = (1,1),merge_mode = mode))
    model.add(Bidirectional(LSTM(100, return_sequences = True),merge_mode = mode))
    model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics =['acc'])
    return model

def create_model(optimizer='adagrad',
                 kernel_initializer='glorot_uniform', 
                 dropout=0.2):
    model = Sequential()
    model.add(InputLayer(input_shape=(1,)))
    model.add(Dense(64,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(1366,activation='sigmoid',kernel_initializer=kernel_initializer))

    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

    return model

