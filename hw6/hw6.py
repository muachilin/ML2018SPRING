from keras import regularizers
import numpy as np
import pandas as pd
import math as math
import keras.backend as K
import sys
import os
import keras
from keras.models import load_model
from keras.layers import Dropout , Flatten
from keras.layers import BatchNormalization,Reshape
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
import string
from keras.layers import MaxPooling1D
from keras.layers import Input,Dot,Add
from keras.layers import ConvLSTM2D
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM,GRU,TimeDistributed
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model


def read_test_data(test_file):
	read_num = 0
	test_data = []
	lines = 0
	with open(test_file , 'r') as f:
		lines = f.readlines()
		for j in range(len(lines)):
			if read_num > 0:
				test_list = lines[j].split(",")
				tmp_list = []
				for i in range(3):
					tmp_list.append(int(test_list[i]))
				test_data.append(tmp_list)
			else:
				read_num = 1
	test_data = np.array(test_data)

	tuser_id = []
	tmovie_id = []
	for i in range(len(test_data)):
		tuser_id.append(test_data[i][1])
		tmovie_id.append(test_data[i][2])
	tuser_id = np.array(tuser_id)
	tmovie_id = np.array(tmovie_id)
	return tuser_id , tmovie_id

def rmse( true_value, prediction_value ):
	tmp = (prediction_value - true_value)
	mean = K.mean( tmp**2 )
	ans = K.sqrt( mean )
	return ans



test_file_name = sys.argv[1]
ans_file_name = sys.argv[2]


model_path = "./0604_model_0.86283.h5"
model = load_model(model_path, custom_objects={'rmse': rmse})
tuser_id , tmovie_id = read_test_data(test_file_name)

prediction = model.predict([tuser_id, tmovie_id])
for i in range(len(prediction)):
	if prediction[i] > 5.0:
		prediction[i] = 5.0
	elif prediction[i] < 1.0:
		prediction[i] = 1.0

ans_file = open(ans_file_name , 'w')
ans_file.write("TestDataID,Rating\n")
for i in range(len(prediction)):
	ans_file.write(str(i+1))
	ans_file.write(",")
	ans_file.write(str(prediction[i][0]))
	ans_file.write("\n")
