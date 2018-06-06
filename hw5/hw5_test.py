from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import numpy as np
import pandas as pd
import math as math
import sys
import os
import keras
from keras.models import load_model
from keras.layers import Dropout , Flatten
from keras.layers import BatchNormalization
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
import string
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM,GRU,TimeDistributed
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from gensim.models.word2vec import Word2Vec


def normal_string(string):
	
	if not string:
		return ""
	if len(string) <= 2:
		return string
	if len(string) > 2 and string[0] == string[1] and string[1] == string[2]:
		return normal_string(string[1:])
	return string[0] + normal_string(string[1:])

def remove_space(text):

	index_list = [i for i, letter in enumerate(text) if letter == '\'']
	remove_list = []
	for i in range(0,len(index_list)):
		if index_list[i]-1 >= 0 and text[index_list[i]-1] == ' ':
			remove_list.append(index_list[i]-1)
		if index_list[i]+1 < len(text) and text[index_list[i]+1] == ' ':
			remove_list.append(index_list[i]+1)
		#remove_list.append(index_list[i])
	text = "".join([char for idx, char in enumerate(text) if idx not in remove_list])
	return text

mode = sys.argv[3]

test_data_filename = sys.argv[1]
t_lines = [line.rstrip('\n') for line in open(test_data_filename,'r' , errors='replace' , encoding='utf-8')]
t_lines = t_lines[1:]
for i in range(0,len(t_lines)):
	num = len(str(i))
	t_lines[i] = t_lines[i][num+1:]
w2v_t_lines = []
for i in range(0, len(t_lines)):
	t_lines[i] = remove_space(t_lines[i])
	tk = text_to_word_sequence(t_lines[i], filters='', lower=True, split=' ')
	tmp_line = []
	tmp = ""
	for j in range(0,len(tk)):
		tk[j] = tk[j].encode("ascii", errors="ignore").decode()
		tk[j] = normal_string(tk[j])
		tmp_line.append(tk[j])
		tmp = tmp + tk[j] + " "
	t_lines[i] = tmp
	w2v_t_lines.append(tmp_line)



model = Word2Vec.load("gensim_w2v_0.82693_0602_model")
word_vectors = model.wv
vocab = []
for k, v in word_vectors.vocab.items():
	vocab.append( (k,v.index) )

vocab = sorted(vocab , key=lambda x:x[1])
word_index_dict = {}
for i in range(0,len(vocab)):
	word = vocab[i][0]
	word_index_dict[word] = i+1
word_index_dict["unknown_word"] = len(vocab)+1

test_ind = []
for i in range(len(w2v_t_lines)):
	tmp = []
	for w in w2v_t_lines[i]:
		if w not in word_index_dict:
			tmp.append(word_index_dict["unknown_word"])
		else:
			tmp.append(word_index_dict[w])
	test_ind.append(tmp)



rnn_model = load_model("0602_gensim_0.82693.h5")
test = sequence.pad_sequences(test_ind, maxlen=33)
p = 0.0
p += rnn_model.predict(test)
ans_filename = sys.argv[2]
ans_file = open(ans_filename , 'w')
ans_file.write("id,label\n")


for i in range(0,len(p)):
	ans_file.write(str(i))
	ans_file.write(',')
	if p[i][0] >= 0.5:
		ans_file.write('1')
	else:
		ans_file.write('0')
	ans_file.write('\n')


