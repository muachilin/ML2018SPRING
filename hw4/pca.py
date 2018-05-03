import sys
import os
import numpy as np
from numpy.linalg import svd
from numpy.linalg import eig
from skimage import io
from skimage import transform



face_folder = sys.argv[1]
if face_folder[len(face_folder)-1] != '/':
	face_folder += "/"
target = sys.argv[2]


image_data = []
for file in os.listdir(face_folder):
	filepath = os.path.join(face_folder , file)
	img = io.imread(filepath)
	img = np.array(img)
	img = img.flatten()
	image_data.append(img)

image_data = np.array(image_data)
image_data_mean = np.mean(image_data,axis=0)

x = image_data - image_data_mean


U, s, V = np.linalg.svd(x.T , full_matrices=False)



target = face_folder + target
ori_img = io.imread(target)

ori_img = np.array(ori_img)
ori_img = np.reshape(ori_img , (1,1080000))
ori_img = ori_img - image_data_mean


weights = np.dot( ori_img , U[:,:4])
recon =  image_data_mean + np.dot(weights, U[:,:4].T)


recon -= np.min(recon)
recon /= np.max(recon)
recon = (recon*255).astype(np.uint8)



io.imsave('reconstruction.png', recon)

