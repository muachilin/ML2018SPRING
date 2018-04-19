import numpy as np
import sys
import pandas as pd
import math as math
import keras
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import ActivityRegularization
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils import np_utils

filename = sys.argv[1]
data = open(filename, 'r', encoding='big5')
dx = pd.read_csv(data,error_bad_lines=False)
dx = dx.ix[:,:]
dx = dx.values[:]
size = 0
for i in range(0,len(dx)):
	dx[i][1] = dx[i][1].split()
	dx[i][1] = list(map(int, dx[i][1]))
	dx[i][1] = np.array(dx[i][1]).astype('float')
	dx[i][1] /=255
	dx[i][1] = np.reshape(dx[i][1],(48,48,1))
dx = np.array(dx)

data = []
for i in range(0,len(dx)):
	data.append(dx[i][1])
data = np.array(data)

label = []
for i in range(0,len(dx)):
	label.append(dx[i][0])
label = np.array(label)
mean, std = np.mean(data, axis=0), np.std(data, axis=0)
data = (data-mean)/std


data_valid = data[15000:]
label_valid = label[15000:]
label = np_utils.to_categorical(label)
label_valid = np_utils.to_categorical(label_valid)

datagen = ImageDataGenerator(
		zoom_range=0.3,
		rotation_range=15,
		width_shift_range=0.15,
		height_shift_range=0.15,
		horizontal_flip=True)

model = Sequential()

model.add(Conv2D(128, kernel_size=(6, 6), input_shape=(48,48,1),padding='same', kernel_initializer='lecun_uniform'))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same',data_format=None))
model.add(Dropout(0.28))

model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))


model.add(SeparableConv2D(256, kernel_size=(5,5), strides=(1, 1), padding='same', data_format=None, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='lecun_normal', pointwise_initializer='lecun_normal', bias_initializer='zeros',))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.35))



model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same',data_format=None))


model.add(Conv2D(512, kernel_size=(5,5), padding='same', kernel_initializer='lecun_normal'))
model.add(ELU(alpha=0.01))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same',data_format=None))
model.add(Dropout(0.52))


model.add(Flatten())
model.add(Dense(256))
model.add(PReLU(alpha_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(0.35))


model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


model.fit_generator(
	datagen.flow(data,label, batch_size=128),
	steps_per_epoch= 800,
	epochs = 400,
	validation_data=(data_valid, label_valid)
	)

model.save('model.h5')
