import sys
import pandas as pd
import numpy as np
import math as math

example = np.load('model.npy')
test_file = sys.argv[1]
te = pd.read_csv(test_file,header = None)
te = te.ix[:,2:]
te = te.values[0:]
testing = []
testing_ori = []
for i in range(0,260):
	testing.append([])
	testing_ori.append([])
for i in range(0,260):
	index = 0
	for j in range(0,18):
		if j != 10 and j!=14 and j!=15:
			for k in range(0,len(te[i*18+j])):
				element = float(te[i*18+j][k])
				if j == 9:
					if element <= 0 or element >= 300:
						if k==0:
							hh = 1
							while float(te[i*18+j][k+hh]) <= 0 or float(te[i*18+j][k+hh]) >= 300:
								hh+=1
								element = float(te[i*18+j][k+hh])
						elif k == len(te[i*18+j])-1:
							element = testing[i][len(testing[i])-1]
						else:
							#element = (float)(testing[i][len(testing[i])-1]+element = testing[i][len(testing[i])-2])*0.5
							element = (float)(testing[i][len(testing[i])-1])
							hh=1
							while float(te[i*18+j][k+hh]) <= 0 or float(te[i*18+j][k+hh]) >= 300:
								hh+=1
							element += float(te[i*18+j][k+hh])
							element = float(element*0.5)
				constant = 1
				testing_ori[i].append(   float(te[i*18+j][k])  )
				testing[i].append(element)
				index += 1
for i in range(len(testing)):
	testing[i].insert(0,1)
testing = np.array(testing)

ans_file = sys.argv[2]
f = open(ans_file,'w')
f.write("id,value\n")

for i in range(0,len(testing)):
	
	ans = np.dot(example,testing[i])
	if ans < 0:
		ans = 1.0
	s = "id_"
	s += str(i)
	s += ","
	s += str(ans)
	s += "\n"
	f.write(s)
