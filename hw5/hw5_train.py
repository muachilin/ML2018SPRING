from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import numpy as np
import pandas as pd
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

tl_file = sys.argv[1]
tn_file = sys.argv[2]

lines = [line.rstrip('\n') for line in open( tl_file ,'r' , errors='replace' , encoding='utf-8')]
n_lines = [line.rstrip('\n') for line in open( tn_file ,'r' , errors='replace' , encoding='utf-8')]

label = []
for i in range(0,len(lines)):
	l = int(lines[i][0])
	label.append(l)
	lines[i] = lines[i][10:]
	lines[i] = lines[i].encode("ascii", errors="ignore").decode()

for i in range(0,len(n_lines)):
	n_lines[i] = n_lines[i].encode("ascii", errors="ignore").decode()


def remove_space(text):
	
	index_list = [i for i, letter in enumerate(text) if letter == '\'']
	remove_list = []
	for i in range(0,len(index_list)):
		if index_list[i]-1 >= 0 and text[index_list[i]-1] == ' ':
			remove_list.append(index_list[i]-1)
		if index_list[i]+1 < len(text) and text[index_list[i]+1] == ' ':
			remove_list.append(index_list[i]+1)
	text = "".join([char for idx, char in enumerate(text) if idx not in remove_list])
	return text
	
w2v_lines = []
for i in range(0, len(lines)):
	lines[i] = remove_space(lines[i])	
	tk = text_to_word_sequence(lines[i], filters='', lower=True, split=' ')
	tmp_line = ""
	tmp_line_w2v = []
	for j in range(0,len(tk)):
		tk[j] = tk[j].encode("ascii", errors="ignore").decode()	
		tk[j] = normal_string(tk[j])
		tmp_line = tmp_line + tk[j] + " "	
		tmp_line_w2v.append(tk[j])
	lines[i] = tmp_line
	w2v_lines.append(tmp_line_w2v)

w2v_n_lines = []

for i in range(0, len(n_lines)):
	n_lines[i] = remove_space(n_lines[i])
	tk = text_to_word_sequence(n_lines[i], filters='', lower=True, split=' ')
	tmp_line = []
	tmp = ""
	for j in range(0,len(tk)):
		tk[j] = tk[j].encode("ascii", errors="ignore").decode()
		tk[j] = normal_string(tk[j])
		tmp = tmp + tk[j] + " "
		tmp_line.append(tk[j])
	n_lines[i] = tmp
	w2v_n_lines.append(tmp_line)	
	

w = w2v_lines + w2v_n_lines

w2v_model = Word2Vec( w , size=350,window=10,
		min_count=5,workers=10)

w2v_model.save("gensim_w2v_0.82693_0602_model")
model = Word2Vec.load("gensim_w2v_0.82693_0602_model")

word_vectors = model.wv

vocab = []
for k, v in word_vectors.vocab.items():
	vocab.append( (k,word_vectors[k],v.index) )

vocab = sorted(vocab , key=lambda x:x[2])
word_index_dict = {}
m_vector_size = model.vector_size
embed_weight_0 = len(vocab)+2
embed_weight = np.zeros((embed_weight_0 , m_vector_size))

del model

for i in range(0,len(vocab)):
	word_index_dict[ vocab[i][0] ] = i+1
	embed_weight[i+1] = vocab[i][1]

embed_weight_shape_0 = embed_weight.shape[0]
embed_weight_shape_1 = embed_weight.shape[1]
embed_weight_add = np.mean(embed_weight[1:-1],axis = 0)
embed_weight[len(vocab)+1] = embed_weight_add
word_index_dict["unknown_word"] = len(vocab)+1
w2v_emb_layer = Embedding(input_dim = embed_weight_shape_0 , output_dim = embed_weight_shape_1,
				weights = [embed_weight] , trainable = False)

train_ind = []
for i in range(len(w2v_lines)):
	tmp = []
	for w in w2v_lines[i]:
		if w not in word_index_dict:
			tmp.append(word_index_dict["unknown_word"])
		else:
			tmp.append(word_index_dict[w])
	train_ind.append(tmp)



w2v = sequence.pad_sequences(train_ind, maxlen=33)
l = label
train = w2v[:180000]
valid = w2v[180000:]
train_label = l[:180000]
valid_label = l[180000:]

model_path = "0602_gensim_0.82693.h5"
checkpoint = ModelCheckpoint(filepath = model_path,verbose=1,save_best_only=True,monitor='val_acc',
			mode='max' )

model = Sequential()
model.add(w2v_emb_layer)

model.add(LSTM(512,recurrent_initializer='glorot_uniform',return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(256,recurrent_initializer='glorot_normal',return_sequences=False))
model.add(Dropout(0.25))

model.add(Dense(256, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.1),activation='relu'))
model.add(Dense(128, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.1),activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit( train , train_label , validation_data=( valid , valid_label )
		, epochs=5, batch_size=128,callbacks=[checkpoint])
model.save(model_path) 


