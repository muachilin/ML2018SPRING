from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
import sys
import numpy as np
import pandas as pd
import math as math

selection = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,83,84,85,86,88,89,91,92,93,95,96,97,98,100,101,102,103,105,106,109,110,111,112,113,114,115,116,117,119,120]

train_X = sys.argv[3]
train_Y = sys.argv[4]

datax = open(train_X, 'r', encoding='big5')
dx = pd.read_csv(datax,error_bad_lines=False)
dx = dx.ix[:,:]
dx = dx.values[:]
matrix = []
ss = []
for i in range(0,len(dx)):
	tmp = []
	for j in range(0,len(dx[i])):
		if j in selection:
			tmp.append(dx[i][j])
	matrix.append(tmp)
	
ss = [0,10,78,79,80]


for j in ss:
	for i in range(0,len(dx)):
		matrix[i].append(dx[i][j]**1.2)
for j in ss:
	for i in range(0,len(dx)):
		matrix[i].append(dx[i][j]**1.4)

datay = open(train_Y, 'r', encoding='big5')
dy = datay.readlines()
label = []
for i in range(0,len(dy)):
	label.append(int(dy[i][0]))

mu_0 = []
mu_1 = []
num_0 = 0
num_1 = 0

test_X = sys.argv[5]

tx = open(test_X, 'r', encoding='big5')
tx = pd.read_csv(tx,error_bad_lines=False)

tx = tx.ix[:,:]
tx = tx.values[:]

tt = []
for i in range(0,len(tx)):
	tmp = []
	for j in range(0,len(tx[i])):
		if j in selection:
			tmp.append(tx[i][j])
	tt.append(tmp)

for j in ss:
	for i in range(0,len(tx)):
		tt[i].append(tx[i][j]**1.2)
for j in ss:
	for i in range(0,len(tx)):
		tt[i].append(tx[i][j]**1.4)

def sigmoidnum(x):
	if x<-100:
		return 0
	elif x > 100:
		return 1
	else:
		return 1 / (1 + math.exp(-x)+1e-10000)
def sigmoid(x):
	ans =  1 / (1 + np.exp(-x))
	
	return ans
def predict_val(x_data,w):
	return sigmoid(np.dot(x_data,w))
def error_val(x_data,y_data,w):
	return predict_val(x_data,w) - y_data
def gra(x_data,y_data,w):
	g = np.zeros(len(w))
	for i in range(0,np.shape(x_data)[0]):
		x_data[i] = np.array(x_data[i])
		e = sigmoidnum(np.dot(np.transpose(w),x_data[i]))
		g += (e - y_data[i])*x_data[i]
	return g/np.shape(x_data)[0]

def training(learning_rate,x_data,y_data):
	w = np.random.randn(np.shape(x_data)[1]) / np.shape(x_data)[1] / np.shape(x_data)[0]
	lam = 1000
	
	l = learning_rate
	g_arr = np.zeros(len(x_data[0]))
	for i in range(0,12000):
		
		g = np.dot(np.transpose(x_data),error_val(x_data,y_data,w))
		g_r = lam * w /len(x_data)
		g_f_r = lam * w /len(x_data)
		g = g + g_r + g_f_r
		loss = -np.mean(y_data * np.log(predict_val(x_data,w) + 1e-20) + (1 - y_data) * np.log((1 - predict_val(x_data,w) + 1e-20)))

		g_arr += g**2
		sigma = np.sqrt(g_arr/(i+1))
		w = w - learning_rate*(1/((i+1)**0.5) * g/sigma)

	return w

mean = np.mean(matrix, axis=0)
std = np.std(matrix, axis = 0)
matrix = (matrix - mean) / (std)

matrix = np.array(matrix)
matrix = np.concatenate((np.ones((matrix.shape[0],1)),matrix), axis=1)

matrix = np.array(matrix)
label = np.array(label)
tt = (tt - mean) / (std)
tt = np.array(tt)
tt = np.concatenate((np.ones((tt.shape[0],1)),tt), axis=1)


example = training(1.2,matrix,label)

prediction = sys.argv[6]

ans = sigmoid(np.dot(tt,example))
f = open(prediction , 'w')
f.write("id,label\n")

for i in range(0,len(ans)):
	s = str(i+1)
	s += ","
	
	if ans[i] >= 0.5:
		s += "1"
	else:
		s+= "0"
	s += "\n"

	f.write(s)

