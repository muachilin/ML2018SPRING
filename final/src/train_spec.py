#This code is using tensorflow backend
#!/usr/bin/env python
# -- coding: utf-8 -- 
import os
import numpy as np
import argparse
import time
import math
import random
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, LSTM, GRU, Dropout, TimeDistributed
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling1D
from keras.layers import Flatten, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History ,ModelCheckpoint
from keras.layers import Activation, LeakyReLU, PReLU


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

def main():
    #training parameters
    batch_size = 128
    epochs = 20

    #load data
    (train_X_v, train_Y_v, train_X_n, train_Y_n) = load_data()
    
    #train cnn
    train(batch_size, epochs, train_X_v, train_Y_v)

def load_data():
    train_X_v = np.load('data/train_X_verified.npy')
    train_Y_v = np.load('data/train_Y_verified.npy')
    train_X_n = np.load('data/train_X_nonveri.npy')
    train_Y_n = np.load('data/train_Y_nonveri.npy')
    print (train_X_v.shape)
    print (train_Y_v.shape)
    #need to reshape data to put in the inputs of Conv2D
    #train_X_v.reshape((163753,128,10,1))
    #train_Y_v.reshape((163753,128,10,1))
    return (train_X_v, train_Y_v, train_X_n, train_Y_n)

def train(batch_size, epochs, train_X_v, train_Y_v): 

    model = Sequential()
    model.add(Conv2D(64,input_shape=(128,10,1), kernel_size=(7, 7), padding='same', kernel_initializer='glorot_normal'))
    model.add(PReLU(shared_axes=[1,2]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(41, activation='softmax', kernel_initializer='glorot_normal'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    filepath = 'model/weights-{epoch:02d}-{val_loss:.2f}.h5'
    hist = History()
    checkpoint1= ModelCheckpoint(filepath,monitor='val_acc', verbose=1,save_best_only=True, mode='max')
    checkpoint2=EarlyStopping(monitor='val_acc',min_delta=0,patience=10,verbose=0, mode='max')
    callbacks_list = [checkpoint1,checkpoint2]

    model.fit(x=train_X_v, y=train_Y_v, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True)

    model.save("model/model.h5")

def predict(model1, model2, test_pixels, prediction_file):
    model_1 = load_model(model1)
    proba_1 = model_1.predict(test_pixels)
    model_2 = load_model(model2)
    proba_2 = model_2.predict(test_pixels)
    proba = proba_1+proba_2
    classes = proba.argmax(axis=-1)   
    with open(prediction_file, 'w') as f:
        f.write('id,label')
        for i in range(len(classes)):
            f.write('\n' + str(i) + ',' + str(classes[i]))


if __name__ == "__main__":
    main()
