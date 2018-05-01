import numpy as np
import sys
import pandas as pd
import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.cluster import KMeans
from keras.optimizers import Adadelta
from skimage.measure import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from keras.models import load_model


image_data_path = sys.argv[1]
image_data = np.load(image_data_path)
for i in range(0,len(image_data)):
	image_data[i] = np.array(image_data[i]).astype('float')/255
image_data = np.array(image_data)

im = []
for i in range(0,len(image_data)):
	im.append(np.array(image_data[i]))
image_data = im
image_data = np.array(image_data)

ori_data = image_data
image_valid = image_data[13800:]
image_data = image_data[:]


e = load_model('0421_2_e_300.h5')
e_image = e.predict(ori_data)

ei = []
for i in range(0,len(e_image)):
        ei.append(np.array(e_image[i]))
e_image = ei
e_image = np.array(e_image)

kmeans = KMeans(n_clusters=2, random_state=0).fit(e_image)


test_case_path = sys.argv[2]

test = open(test_case_path , 'r', encoding='big5')
test = pd.read_csv(test,error_bad_lines=False)
test = test.ix[:,:]
test = test.values[:]
test_data = []
for i in range(0,len(test)):
	d = []
	for j in range(0,len(test[i])):
		if j>0:
			d.append(int(test[i][j]))
	test_data.append(d)

test_data = np.array(test_data)

ans_path = sys.argv[3]

ans_file = open(ans_path,'w')
ans_file.write('ID,Ans\n')
for i in range(0,len(test_data)):

	p1 = kmeans.labels_[test_data[i][0]]
	p2 = kmeans.labels_[test_data[i][1]]
	s = ""
	s += str(i)
	s += ","
	if p1 == p2:
		s += "1"
	else:
		s += "0"
	s += "\n"
	ans_file.write(s)



