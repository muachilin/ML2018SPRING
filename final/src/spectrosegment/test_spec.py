import numpy as np
from sys import argv
import os
from keras.models import load_model
import pickle
import csv
import tensorflow as tf
from sklearn.utils.class_weight import compute_sample_weight


win_size = 20
hop_size = 10

def load_pk(test):
    with open(test,'rb') as f:
        test_X = pickle.load(f)
    return test_X

def add_csv(csv_file, name, prediction,reverse_label_dict):
    rank = np.argsort(-prediction,axis=1)    
    with open(csv_file,'a') as f:
        f.write(name + ',' + reverse_label_dict[rank[0,0]] + ' ' + reverse_label_dict[rank[0,1]] + ' ' + reverse_label_dict[rank[0,2]] + '\n')
def read_test(test_csv):
    test_order = []
    with open(test_csv,newline='') as csvfile:
        test = csv.reader(csvfile)
        for row in test:
            if row[0] != 'fname': test_order.append(row[0])
    return test_order

def main():
    #model_path = argv[1]
    test_path = argv[1]
    test_csv = argv[2]
    out_csv = argv[3]
    
    print('===load model===')
    model = []
    model.append(load_model('model/retrain07-00-0.919.h5'))
    model.append(load_model('model/retrain11-00-0.920.h5'))
    model.append(load_model('model/retrain12-01-0.933.h5'))
    model.append(load_model('model/retrain13-00-0.931.h5'))
    model.append(load_model('model/retrain21-00-0.931.h5'))
    model.append(load_model('model/retrain25-00-0.909.h5'))
    model.append(load_model('model/retrain25-00-0.932.h5'))
    model.append(load_model('model/retrain26-01-0.968.h5'))
    model.append(load_model('model/retrain27-00-0.920.h5'))
    print('===load test data===')
    test_X = load_pk(test_path)
    print('===load test order===')
    test_order = read_test(test_csv)
    print('===write csv===')
    with open(out_csv,'w') as f:
        f.write('fname,label\n')
    
    label_dict = load_pk('label_dict.pkl')
    reverse_label_dict = dict(zip(label_dict.values(),label_dict.keys()))
    total_len = len(test_X)
    i = 1
    print('===predicting===')
    for name in test_order:
        print(str(i) + '/' + str(total_len))
        i = i + 1
        if name == '0b0427e2.wav' or name == '6ea0099f.wav' or name == 'b39975f5.wav':
            add_csv(out_csv, name, np.zeros((1,41)), reverse_label_dict)
            continue        
        audio = np.expand_dims(test_X[name], axis=3)
        prediction = np.zeros((1,41))
        cnt = 0
        head = 0
        tail = win_size     
        for k in range(len(model)):
            prediction += np.sum(model[k].predict(audio), axis=0)
            cnt += 1
        if cnt != 0:
            prediction = prediction / cnt
        add_csv(out_csv, name, prediction, reverse_label_dict)
if __name__ == '__main__':
    main()
