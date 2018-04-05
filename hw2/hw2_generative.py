from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
import sys
import numpy as np
import pandas as pd
import math as math

train_X = sys.argv[3]
train_Y = sys.argv[4]

datax = open(train_X, 'r', encoding='big5')
dx = pd.read_csv(datax,error_bad_lines=False)
dx = dx.ix[:,:]
dx = dx.values[:]
matrix = []
for i in range(0,len(dx)):
	tmp = []
	for j in range(0,len(dx[i])):
		tmp.append(dx[i][j])
	matrix.append(tmp)

datay = open(train_Y, 'r', encoding='big5')
dy = datay.readlines()
label = []
for i in range(0,len(dy)):
	label.append(int(dy[i][0]))

mu_0 = []
mu_1 = []
num_0 = 0
num_1 = 0

mean = np.mean(matrix, axis=0)
std = np.std(matrix, axis=0)
matrix = (matrix - mean) / (std)

test_X = sys.argv[5]
tx = open(test_X, 'r', encoding='big5')
tx = pd.read_csv(tx,error_bad_lines=False)

tx = tx.ix[:,:]
tx = tx.values[:]


tt = []
for i in range(0,len(tx)):
	tmp = []
	for j in range(0,len(tx[i])):	
		tmp.append(tx[i][j])
	tt.append(tmp)

tt = np.array(tt)
tt =  (tt - mean) / (std)
g0 = []
g1 = []
for i in range(len(matrix)):
	if label[i]==0:
		g0.append(matrix[i])
		num_0 += 1
		if num_0 == 1:
			mu_0 = matrix[i]
			mu_0 = np.array(mu_0)
		else:
			mu_0 += np.array(matrix[i])
	elif label[i]==1:
		g1.append(matrix[i])
		num_1 += 1
		if num_1 == 1:
			mu_1 = matrix[i]
			mu_1 = np.array(mu_1)
		else:
			mu_1 += np.array(matrix[i])
mu0 = [[]]
mu1 = [[]]
for i in range(0,len(mu_0)):
	m = mu_0[i]
	mu0[0].append(m/num_0)
	m = mu_1[i]
	mu1[0].append(m/num_1)
mu0 = np.array(mu0)
mu1 = np.array(mu1)

cov_0 = []
cov_1 = []
index0 = 0
index1 = 0

for i in range(0,len(matrix)):
	mm = []
	mm.append(matrix[i])
	if label[i]==0:
		t = np.array(mm) - np.array(mu0)	
		tmp = np.dot(np.transpose(t),t)
		
		if index0 == 0:
			index0 += 1
			cov_0 = tmp
			cov_0 = np.array(cov_0)
		else:
			index0 += 1
			cov_0 += np.array(tmp)
	elif label[i]==1:
		t = np.array(mm) - np.array(mu1)
		
		tmp = np.dot(np.transpose(t),t)
		
		if index1 == 0:
			index1 += 1
			cov_1 = tmp
			cov_1 = np.array(cov_1)
		else:
			index1 += 1
			cov_1 += np.array(tmp)

cc = []

cc = cov_1

covariance = []
for i in range(0,len(cov_0)):
	tmp = []
	for j in range(0,len(cov_0[0])):
		a = cov_0[i][j]
		b = cov_1[i][j]
		tmp.append( (a+b)/(num_0+num_1) )
	covariance.append(tmp)		

cov_0/=num_0
cov_1/=num_1

covariance = np.array(covariance)


def gaussian(muu,cov,x):
	
	a = np.array(x) - np.array(muu)
	
	e1 = np.dot(np.dot(a,pinv(cov)),np.transpose(a))	
	
	e1 *= (-1/2)
	
	etmp = (math.e)**(e1)
	eigen,v = np.linalg.eig(cov)
	aa = 1
	
	
	for ee in eigen:	
		if ee > 1e-12:
			aa *=ee

	f = (1/((2*math.pi)**(len(x)/2)))/((aa)**(1/2))
	
	f = f*etmp

	return f

def egaussian(mean,cov,x):
	t = -1 / 2 * np.dot(np.dot((x - mean), np.linalg.inv(cov)), (x - mean).T)
	return 1 / (((np.linalg.det(cov)))**0.5) * np.exp(t)


ans = sys.argv[6]
f = open(ans,'w')
f.write("id,label\n")
for i in range(0,len(tt)):
	s = str(i+1)
	s += ","
	f0 = gaussian(mu0,covariance,tt[i])
	f1 = gaussian(mu1,covariance,tt[i])
	ans = (f0*num_0+1e-1000)/(f0*num_0+f1*num_1+1e-1000)
	if ans >= 0.5:
		s += "0"
	else:
		s+= "1"
	s += "\n"

	f.write(s)

