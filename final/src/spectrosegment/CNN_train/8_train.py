import numpy as np
from sys import argv
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import PReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#set_session(sess)


def cnn_model():
    model = Sequential()
    model.add(Conv2D(100, (10, 3), input_shape = (128, 10, 1), padding = 'same') )
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2))) # 64 * 5
    model.add(Dropout(0.3)) 

    model.add(Conv2D(200, (10, 3), padding = 'same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2))) # 32 * 2
    model.add(Dropout(0.35))
    
    model.add(Conv2D(250, (10, 3), padding = 'same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2))) # 32 * 2
    model.add(Dropout(0.35))

    
    model.add(Flatten())
    
    model.add(Dense(units = 200, activation = 'relu'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(units = 100, activation = 'relu'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
  
  
    model.add(Dense(units = 41, activation = 'softmax'))
    model.summary()
    return model

def train(modelname):
    train_X0 = np.load('train_X_verified.npy')
    train_Y0 = np.load('train_Y_verified.npy')
    train_X0 = train_X0.reshape(train_X0.shape[0],128,10,1)
    train_Y0 = train_Y0.reshape(train_Y0.shape[0],train_Y0.shape[1])
    #train_X1 = np.load('train_X_nonveri.npy')
    #train_Y1 = np.load('train_Y_nonveri.npy')
    #train_X1 = train_X1.reshape(train_X1.shape[0],128,10,1)
    #train_Y1 = train_Y1.reshape(train_Y1.shape[0],train_Y1.shape[1])
    #train_X = np.concatenate((train_X0[:81876],train_X1[:300000]), axis=0)
    #train_Y = np.concatenate((train_Y0[:81876],train_Y1[:300000]), axis=0)
    

    np.random.seed(1200)
    index = np.random.permutation(len(train_X0))
    
    train_X0 = train_X0[index]
    train_Y0 = train_Y0[index]
    train_X = train_X0[14000:]
    train_Y = train_Y0[14000:]
    val_X = train_X0[:14000]
    val_Y = train_Y0[:14000]
    
    
    model = cnn_model()
    checkpoint =[ModelCheckpoint('models/'+modelname,  # model filename
                                            monitor='val_loss', # quantity to monitor
                                            verbose=1, # verbosity - 0 or 1
                                            save_best_only= True, # The latest best model will not be overwritten
                                            mode='auto'), # The decision to overwrite model is made 
                            EarlyStopping(monitor = 'val_loss',
                                            patience = 20,
                                            verbose = 0)]

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(train_X,train_Y,epochs=150,batch_size=128,validation_data=(val_X,val_Y),verbose=1,callbacks=checkpoint)

def retrain(modelname):
    train_X0 = np.load('train_X_verified.npy')
    train_Y0 = np.load('train_Y_verified.npy')
    train_X0 = train_X0.reshape(train_X0.shape[0],128,10,1)
    train_Y0 = train_Y0.reshape(train_Y0.shape[0],train_Y0.shape[1])
    train_X1 = np.load('X_nonveri_filtered.npy')
    train_Y1 = np.load('Y_nonveri_filtered.npy')
    train_X1 = train_X1.reshape(train_X1.shape[0],128,10,1)
    train_Y1 = train_Y1.reshape(train_Y1.shape[0],train_Y1.shape[1])
    total_X = np.concatenate((train_X0,train_X1), axis=0)
    total_Y = np.concatenate((train_Y0,train_Y1), axis=0)
    total_len = total_X.shape[0]
    print(total_Y.shape)
    print(total_X.shape)
    print(total_len)
    
    model = cnn_model()
    checkpoint =[ModelCheckpoint('models/'+modelname,  # model filename
                                            monitor='val_loss', # quantity to monitor
                                            verbose=0, # verbosity - 0 or 1
                                            save_best_only= True, # The latest best model will not be overwritten
                                            mode='auto'), # The decision to overwrite model is made 
                            EarlyStopping(monitor = 'val_loss',
                                            patience = 20,
                                            verbose = 0)]

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(total_X[int(np.floor(total_len*0.1)):],total_Y[int(np.floor(total_len*0.1)):],epochs=100,batch_size=128,validation_data=(total_X[:int(np.floor(total_len*0.1))],total_Y[:int(np.floor(total_len*0.1))]),verbose=1,callbacks=checkpoint)

def main():
    modelname = argv[1]
    if argv[2] == 'train':
        train(modelname)
    elif argv[2] == 'retrain':
        retrain(modelname)
if __name__ == '__main__':
    main()
