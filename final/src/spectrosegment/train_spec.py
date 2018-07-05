import numpy as np
from sys import argv
import os
from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.utils.class_weight import compute_sample_weight


def cnn_model():
    model = Sequential()
    #VGG-16
    input_shape = (128,20,1)
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    """
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    """
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(41, activation='softmax'))
   
    model.summary()
    return model

def train(modelname):
    train_X0 = np.load('train_X_verified20.npy')
    train_Y0 = np.load('train_Y_verified20.npy')
    train_X0 = train_X0.reshape(train_X0.shape[0],128,20,1)
    train_Y0 = train_Y0.reshape(train_Y0.shape[0],train_Y0.shape[1])

    np.random.seed(0)
    cv_idx_tmp = np.random.permutation(68059)
    cv_fold_1 = cv_idx_tmp[0:6817]
    cv_fold_2 = cv_idx_tmp[6817:68059]   
    train_X = train_X0[cv_fold_2[:,],:]
    train_Y = train_Y0[cv_fold_2[:,],:]
    val_X = train_X0[cv_fold_1[:,],:]
    val_Y = train_Y0[cv_fold_1[:,],:]
    
    print(train_X.shape)
    print(train_Y.shape)
    model = cnn_model()
    #save_path = '../model/train28-{epoch:02d}-{val_acc:.3f}-20.h5'   
    checkpoint =[ModelCheckpoint(save_path = modelname,  # model filename
                                 monitor='val_acc', # quantity to monitor
                                 verbose = 1, # verbosity - 0 or 1
                                 save_best_only= True, # The latest best model will not be overwritten
                                 mode='max'), # The decision to overwrite model is made 
                   EarlyStopping(monitor = 'val_loss',
                                 patience = 10,
                                 verbose = 1)]
    
    #y_integers = np.argmax(train_Y, axis=1)
    #sample_weights = compute_sample_weight('balanced', y_integers) #sample_weights is a 63517 length vector
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(train_X,train_Y,epochs=100,batch_size=512,validation_data=(val_X,val_Y),
            verbose=1,callbacks=checkpoint)
    model.save('../model/'+modelname)

def retrain(modelname):
    train_X0 = np.load('train_X_verified20.npy')
    train_Y0 = np.load('train_Y_verified20.npy')
    train_X0 = train_X0.reshape(train_X0.shape[0],128,20,1)
    train_Y0 = train_Y0.reshape(train_Y0.shape[0],train_Y0.shape[1])
    train_X1 = np.load('X_nonveri_filtered.npy')
    train_Y1 = np.load('Y_nonveri_filtered.npy')
    train_X1 = train_X1.reshape(train_X1.shape[0],128,20,1)
    train_Y1 = train_Y1.reshape(train_Y1.shape[0],train_Y1.shape[1])
    total_X = np.concatenate((train_X0,train_X1), axis=0)
    total_Y = np.concatenate((train_Y0,train_Y1), axis=0)
    total_len = total_X.shape[0]
    print(total_Y.shape)
    print(total_X.shape)
    print(total_len)
    
    model = cnn_model()
    #model.load_weights(modelname)
    model = load_model(modelname)
    #save_path = '../model/train27-{epoch:02d}-{val_acc:.3f}-20.h5' 
    checkpoint = [ModelCheckpoint(save_path=modelname,  # model filename
                                 monitor='val_acc', # quantity to monitor
                                 verbose=1, # verbosity - 0 or 1
                                 save_best_only= True, # The latest best model will not be overwritten
                                 mode='max'), # The decision to overwrite model is made 
                   EarlyStopping(monitor = 'val_loss',
                                 patience = 5,
                                 verbose = 1)]
    
    #y_integers = np.argmax(total_Y[int(np.floor(total_len*0.1)):], axis=1)
    #sample_weights = compute_sample_weight('balanced', y_integers)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(total_X[int(np.floor(total_len*0.1)):],total_Y[int(np.floor(total_len*0.1)):],
            validation_data=(total_X[:int(np.floor(total_len*0.1))],total_Y[:int(np.floor(total_len*0.1))]),
            verbose=1,callbacks=checkpoint,epochs=100,batch_size=512)

def main():
    modelname = argv[1]
    if argv[2] == 'train':
        train(modelname)
    elif argv[2] == 'retrain':
        retrain(modelname)
if __name__ == '__main__':
    main()
