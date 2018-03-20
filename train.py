import sys
import pandas as pd
import numpy as np
import math as math
data = open('./data/train.csv', 'r', encoding='big5')
df = pd.read_csv(data,header = None)
df = df.ix[:,2:]
df = df.values[1:]
matrix = []
for i in range(0,len(df)):
	array = []
	w = i%18
	if w!=10 and w!= 14 and w!=15:
		for j in range(0,len(df[i])):
			array.append((df[i][j]))	
		matrix.append(array)
label_array=[]
training_data = []
for i in range(0,15):
	label_array.append(matrix[i][0])
	training_data.append([])

for i in range(0,len(matrix)):
	tmp = matrix[i][1:]
	for j in range(0,len(tmp)):
		tmp[j] = float(tmp[j])
	training_data[i%15].append(tmp)

training = []
for i in range(0,len(training_data)):
	tmp = []
	for j in range(0,len(training_data[i])):
		for k in range(0,len(training_data[i][j])):
			tmp.append(training_data[i][j][k])
	training.append(tmp)

y_array=[]
index = 1
training_data = []

for i in range(0,5751):
	if i >= 24*20*index - 9 and i < 24*20*index:
		continue
	if i == 24*20*index:
		index+=1
	tmp = []
	if training[0][i] == 0 and training[0][i+1] == 0:
		continue
	qq = True
	for k in range(0,len(training)):
		for h in range(0,9): 
			element = training[k][i+h]
			if k==9:
				if element <= 0 or element >= 300:
					qq = False
					if h == 0:
						hh =1
						while training[k][i+h+hh] <= 0 or training[k][i+h+hh] >= 300:
							hh+=1
						element = training[k][i+h+hh]
					elif h == 8:
						element = tmp[len(tmp)-1]
					else:
						element = tmp[len(tmp)-1]
						hh =1
						while training[k][i+h+hh] <= 0 or training[k][i+h+hh] >= 300:
							hh+=1
						element += training[k][i+h+hh]
						element = (float)(element*0.5)
			tmp.append(element)
	if qq == True:
		training_data.append(tmp)
		y_array.append(training[9][i+9])
mean_array = []
std_array = []

def predict_val(x_data,w):
	return np.dot(x_data,w)
def error_val(x_data,y_data,w):
	return predict_val(x_data,w) - y_data
def training(weight_array,learning_rate,gradient_array,t,x_data,y_data,b):
	
	w = np.random.randn(np.shape(x_data)[1]) / np.shape(x_data)[1] / np.shape(x_data)[0]
	w[89] = w[89]*10
	w[90] = w[90]*500	
	lam = 100
	g_arr = np.zeros(len(x_data[0]))
	
	for i in range(0,5000):
		loss = np.sqrt(np.sum( (error_val(x_data,y_data,w))**2 ) / len(x_data)) 
		g = np.dot(np.transpose(x_data),error_val(x_data,y_data,w))
		g_r = lam * w /len(x_data)
		g_f_r = lam * w /len(x_data)
		g = g + g_r + g_f_r
		g_arr += g**2
		sigma = np.sqrt(g_arr/(i+1))
		w = w - learning_rate*(1/((i+1)**0.5) * g/sigma)
		

		
	return w

for i in range(len(training_data)):
	training_data[i].insert(0,1)
training_data = np.array(training_data)

initial_weight_array = np.zeros(len(training_data[0]))
initial_gradient = []
result_array = []
example = training(initial_weight_array,0.9,initial_gradient,0,training_data,y_array,0.0)
np.save('model.npy', example)
