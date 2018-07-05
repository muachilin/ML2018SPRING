import numpy as np
from sys import argv
import os
from keras.models import load_model
import pickle
import csv

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

def filternonveri(result,Y,X):
    nonveri_X = []
    nonveri_Y = []
    for idx, r in enumerate(result):
        rank = np.argsort(-r,axis=0)
        label = np.argmax(Y[idx])
        #print(rank)
        #print(label)
        if label in rank[:1]:
            nonveri_Y.append(Y[idx])
            nonveri_X.append(X[idx])
        #if idx == 3: break
    nonveri_X = np.stack(nonveri_X)
    nonveri_Y = np.stack(nonveri_Y)
    print(nonveri_X.shape)
    print(nonveri_Y.shape)
    np.save('X_nonveri_filtered.npy',nonveri_X)
    np.save('Y_nonveri_filtered.npy',nonveri_Y)

def read_test(test_csv):
    test_order = []
    with open(test_csv,newline='') as csvfile:
        test = csv.reader(csvfile)
        for row in test:
            if row[0] != 'fname': test_order.append(row[0])
    return test_order

def main():
    model_path = argv[1]
    nonveri_X = argv[2]
    nonveri_Y = argv[3]
    #out_csv = argv[4]
    
    print('===load model===')
    model = load_model(model_path)
    print('===load test data===')
    X = np.load(nonveri_X)
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    Y = np.load(nonveri_Y)
    print('===load test order===')
    print('===write csv===')
    total_len = X.shape[0]
    print('===predicting===')
    result = model.predict(X,verbose=1)
    print(result.shape)
    print(Y.shape)
    filternonveri(result,Y,X)

if __name__ == '__main__':
    main()
