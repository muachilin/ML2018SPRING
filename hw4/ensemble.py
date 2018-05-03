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

#0430night.h5
#0501night.h5

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
	meanfile = "cnn_mean_0409_520.npy"
	stdfile = "cnn_std_0409_520.npy"

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


model_0 = load_model('0409_model_520.h5')
model_1 = load_model('0430night.h5')
model_2 = load_model('0501night.h5')
model_3 = load_model('0416_model_400.h5')
model_4 = load_model('0413_model_520.h5')

p_0 = 0.00
p_0 += model_0.predict(test)
prediction_0 = np.argmax(p_0, axis=-1)

p_1 = 0.00
p_1 += model_1.predict(test)
prediction_1 = np.argmax(p_1, axis=-1)

p_2 = 0.00
p_2 += model_2.predict(test)
prediction_2 = np.argmax(p_2, axis=-1)

p_3 = 0.00
p_3 += model_3.predict(test)
prediction_3 = np.argmax(p_3, axis=-1)


p_4 = 0.00
p_4 += model_4.predict(test)
prediction_4 = np.argmax(p_4, axis=-1)


f = open(ansfile,'w')
f.write("id,label\n")
for i in range(0,len(prediction_0)):
	f.write(str(i)+",")
	tmp = []
	tmp.append(prediction_0[i])
	tmp.append(prediction_1[i])
	tmp.append(prediction_2[i])
	tmp.append(prediction_3[i])
	tmp.append(prediction_4[i])
	tmp = np.array(tmp)
	counts = np.bincount(tmp)
	c = np.argmax(counts)
	f.write(str(c))
	f.write("\n")


