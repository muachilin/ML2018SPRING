import numpy as np
from sys import argv
import librosa
from os import listdir
from os.path import isfile, join
import pickle


def segment(spectro):
    test_segs = []
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

    while (tail <= audio_len):
        test_segs.append(spectro[:,head:tail])
        head = head + hop_size
        tail = tail + hop_size
    
    return np.stack(test_segs)

def main():
    direc = argv[1]
    output_name = argv[2]
	
    test_X = {}
    filelist = [f for f in listdir(direc) if isfile(join(direc,f))]
    i = 0
    print('Scan Files and compute spectrograms')
    for f in filelist:
        print(str(i) + '/' + str(len(filelist)))
        print(f)
        if f == '0b0427e2.wav' or f == '6ea0099f.wav' or f == 'b39975f5.wav': continue
        
        i = i + 1
        y,sr = librosa.load(direc + '/' + f)
        spectro = librosa.feature.melspectrogram(y=y,sr=sr)
        print(spectro.shape)

        testsegs = segment(spectro)
        test_X[f] = testsegs
        print(testsegs.shape)
    
    print('Start saving')
    pickle.dump(test_X,open(output_name,'wb'))

if __name__ == '__main__':
    main()
