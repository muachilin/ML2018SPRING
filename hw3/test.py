import numpy as np
import sys
import pandas as pd
import math as math
import keras
from keras.layers import AveragePooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.advanced_activations import *
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import load_model

testfile = sys.argv[1]
ansfile = sys.argv[2]
modelfile = ""
meanfile = ""
stdfile = ""
if sys.argv[3] == "public":
	modelfile = "./model/0409_model_520.h5"
	meanfile = "cnn_mean_0409_520.npy"
	stdfile = "cnn_std_0409_520.npy"
else:
	modelfile = "./model/0416_model_400.h5"
	meanfile = "cnn_mean_0416_400.npy"
	stdfile = "cnn_std_0416_400.npy"

test = open(testfile, 'r', encoding='big5')
t = pd.read_csv(test,error_bad_lines=False)
t = t.ix[:,:]
t = t.values[:]
for i in range(0,len(t)):
	t[i][1] = t[i][1].split()
	t[i][1]=list(map(int, t[i][1]))
	t[i][1] = np.array(t[i][1]).astype('float')
	t[i][1] /=255
	t[i][1] = np.reshape(t[i][1],(48,48,1))
t = np.array(t)

test = []
for i in range(0,len(t)):
        test.append(t[i][1])
test = np.array(test)


mean = np.load(meanfile)
std = np.load(stdfile)


test = (test-mean)/std


model = load_model(modelfile)


p = 0.00
p += model.predict(test)
prediction = np.argmax(p, axis=-1)


f = open(ansfile,'w')
f.write("id,label\n")
for i in range(0,len(prediction)):
	f.write(str(i)+",")
	f.write(str(prediction[i]))
	f.write("\n")


