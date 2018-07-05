'''
    Usage
    python3 preprocess_train.py path/to/train.csv path/to/audio_train
'''
import numpy as np
from sys import argv
import librosa
from os import listdir
from os.path import isfile, join
import pickle
import collections
''' Extract verified data or not. '''
VERIFIED = 0 
audiolen_histogram = collections.Counter()
label_histogram = collections.Counter()

def save_dict(dictionary, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

def preprocess_label(file):
    ''' 
    Process labels to a dictionary while key is name of label ('Saxophone') and value is index.
    The indices represents which dimension of the label's one hot vector will be 1.	

    Input
	train.csv
	
    Returns
	label_dict
    '''
    label_dict = {}
    with open(file,'r') as f:
        i = 0
        for line in f:
            print(i)
            sample = line.split(',')
            label_histogram[sample[1]] += 1
            if sample[1] not in label_dict and sample[1] != 'label':
                label_dict[sample[1]] = i
                i = i + 1
    save_dict(label_dict, 'label_dict')
    return label_dict 

def onehot_label(label_dict, label):
	'''
	Input
		label_dict: key

	'''
	onehot = np.zeros((len(label_dict),1))
	idx = label_dict[label]
	onehot[idx,0] = 1
	return onehot

def fname_label(file):
	map = {}
	label_dict = preprocess_label(file)
	with open(file,'r') as f:
		for line in f:
			
			sample = line.split(',')

			if VERIFIED:				
				if sample[2] == '1\n':
					onehot = onehot_label(label_dict, sample[1])
					map[sample[0]] = onehot
					
			else:
				if sample[2] == '0\n':
					onehot = onehot_label(label_dict, sample[1])
					map[sample[0]] = onehot
	return map

def audio_label(direc, map):
    train_X = []
    train_Y = []
    filelist = [f for f in listdir(direc) if isfile(join(direc,f))]
    i = 1
    print('Scan Files, Compute Spectrograms and Segment')
    for f in filelist:
        print(str(i) + '/' + str(len(filelist)-1))
        i = i + 1
        try:
            label = map[f]
        except:
            continue
        y,sr = librosa.load(direc + '/' + f)
        spectro = librosa.feature.melspectrogram(y=y,sr=sr)
        train_X, train_Y = segment(spectro,label,train_X,train_Y)

    print('Stacking')
    train_X = np.stack(train_X)
    train_Y = np.stack(train_Y)
    print(train_X.shape)
    print(train_Y.shape)
    ''' 
    np.random.seed(1234)
    cv_idx_tmp = np.random.permutation(68059)
    cv_fold_1 = cv_idx_tmp[:int(0.1*len(train_X))]
    cv_fold_2 = cv_idx_tmp[int(0.1*len(train_X)):]
    train_X0 = train_X[cv_fold_2[:,],:,:]
    train_Y0 = train_Y[cv_fold_2[:,],:,:]
    val_X0 = train_X[cv_fold_1[:,],:,:]
    val_Y0 = train_Y[cv_fold_1[:,],:,:]

    print(val_X0.shape)
    print(val_Y0.shape)
    
    tmp_X0, tmp_Y0 = mixup(train_X0,train_Y0,alpha=1)
    train_X0, train_Y0 = np.r_[train_X0, tmp_X0], np.r_[train_Y0, tmp_Y0]
    print(train_X0.shape)
    print(train_Y0.shape)
    '''

    if VERIFIED:
        np.save('train_X_verified.npy', train_X0)
        np.save('train_Y_verified.npy', train_Y0)
        np.save('val_X_verified.npy', val_X0)
        np.save('val_Y_verified.npy', val_Y0)
    else:
        np.save('train_X_nonveri.npy', train_X)
        np.save('train_Y_nonveri.npy', train_Y)
    

def segment(spectro, label, train_X, train_Y):
    win_size = 20
    hop_size = 10
    audio_len = spectro.shape[1]
    head = 0
    tail = win_size
    if audio_len < win_size:
        tmp = spectro
        tmp_len = audio_len
        while audio_len < win_size:
            spectro = np.concatenate((spectro,tmp),axis=1)
            audio_len += tmp_len
         
    audiolen_histogram[audio_len] += 1
    while (tail <= audio_len):
        train_X.append(spectro[:,head:tail])
        train_Y.append(label)
        head = head + hop_size
        tail = tail + hop_size
    return train_X, train_Y

def mixup(data, one_hot_labels, alpha=1, debug=False):
    np.random.seed(42)

    batch_size = len(data)
    weights = np.random.beta(alpha, alpha, batch_size)
    index = np.random.permutation(batch_size)
    x1, x2 = data, data[index]
    x = np.array([x1[i] * weights[i] + x2[i] * (1 - weights[i]) for i in range(len(weights))])
    y1 = one_hot_labels.astype(np.float)
    y2 = one_hot_labels[index].astype(np.float)
    y = np.array([y1[i] * weights[i] + y2[i] * (1 - weights[i]) for i in range(len(weights))])
    if debug:
        print('Mixup weights', weights)
    return x, y

def main():
    train = argv[1]
    direc = argv[2]
    map = fname_label(train)
    audio_label(direc, map)
    print(label_histogram)
    statcounter = collections.Counter()
    
    totallen = 0
    filelen = 0
    for key in audiolen_histogram:
        if key <= 20:
            statcounter[20] += 1
        if key <= 30:
            statcounter[30] += 1
        if key <= 40:
            statcounter[40] += 1
        if key <= 50:
            statcounter[50] += 1
        totallen += key * audiolen_histogram[key]
        filelen += audiolen_histogram[key]
    
    print('total: ' + str(sum(audiolen_histogram.values())) )
    print('average length: ' + str(totallen/filelen))
    

if __name__ == '__main__':
	main()
