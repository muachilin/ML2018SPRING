import numpy as np
import os, argparse 
import re
import pickle
from librosa.core import load
from librosa.feature import mfcc
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D,GlobalAveragePooling2D,Dense,Activation,MaxPooling2D,Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import History ,ModelCheckpoint
from keras.models import load_model

def main():
    """
    usage:
    -to train: python3 train_mfcc.py train --mfcc_path data/mfcc.txt
    -to test: python3 train_mfcc.py test --mfcc_path data/test_mfcc.txt
    
    required directory and files:
    -/audio_train
    -/audio_test
    -/data/train.csv
    """
    parser = argparse.ArgumentParser(description='Sound classification')
    parser.add_argument('action', choices=['train','test'])

    # training argument
    parser.add_argument('--mfcc_path', default='data/mfcc.txt', type=str)
    
    args = parser.parse_args()
    
    dic={"Acoustic_guitar"  :0 , "Applause"             :1 , "Bark"         :2 , "Bass_drum"            :3 , "Burping_or_eructation"    :4 ,
        "Bus"               :5 , "Cello"                :6 , "Chime"        :7 , "Clarinet"             :8 , "Computer_keyboard"        :9 ,
        "Cough"             :10, "Cowbell"              :11, "Double_bass"  :12, "Drawer_open_or_close" :13, "Electric_piano"           :14,
        "Fart"              :15, "Finger_snapping"      :16, "Fireworks"    :17, "Flute"                :18, "Glockenspiel"             :19,
        "Gong"              :20, "Gunshot_or_gunfire"   :21, "Harmonica"    :22, "Hi-hat"               :23, "Keys_jangling"            :24,
        "Knock"             :25, "Laughter"             :26, "Meow"         :27, "Microwave_oven"       :28, "Oboe"                     :29,
        "Saxophone"         :30, "Scissors"             :31, "Shatter"      :32, "Snare_drum"           :33, "Squeak"                   :34,
        "Tambourine"        :35, "Tearing"              :36, "Telephone"    :37, "Trumpet"              :38, "Violin_or_fiddle"         :39, "Writing":40}
    
    if args.action == 'train':
        load_data(dic) #yield 'data/mfcc.txt' and 'data/label.npy'
        normalize() #yield 'data/para.npy'
        padded_data(args.mfcc_path) #yield 'data/paddeddata.npy'
        train() #yield 'model/many_models.h5'
    else:
        load_test_data(dic) #yield 'data/test_mfcc.txt'
        padded_data(args.mfcc_path) #yield 'data/test_paddeddata.npy'
        predict(dic) #yield 'result.csv'

def load_data(dic):

    ""
    #讀進音檔，每一個音檔用mfcc轉成(40,l)的nparray(l與音檔長度有關)
    #依照train.csv的順序存成一個list並存成mfcc.txt
    #把每種label轉成數字，依照train.csv的順序存成label.npy
    ""
    data=[]
    label=np.zeros((9473,1))
    
    file=open("data/train.csv","r").readlines()

    for i in range(1,len(file)):
        row = re.split(",|\n|\t", file[i])
        label[i-1]=dic[row[1]]
        wav,sr=load("audio_train/"+row[0])
        if row[0] in ["0b0427e2.wav","6ea0099f.wav","b39975f5.wav"]:
            data.append(np.zeros((40,1)))
            continue
        #wav, sr = load("/bin/data/audio_test/" + row[0])
        data.append(mfcc(y=wav, sr=sr,n_mfcc=40))
        if i%20==0:
            print(i)

    with open("data/mfcc.txt", "wb") as fp:
        pickle.dump(data, fp)
    np.save("data/label.npy",label)

def load_test_data(dic):
    path = 'data/audio_test'
    data = []
    empty = []
    for file in os.listdir(path):
        wav, sr = load(os.path.join(path, file))
        if len(wav)!=0:
            print('mfcc sound...', file)
            data.append(mfcc(y=wav, sr=sr, n_mfcc=40))
        else:
            data.append(np.zeros((40,1)))
            empty.append(file)
    print(empty)
    with open('data/test_mfcc.txt', 'wb') as fp:
        pickle.dump(data, fp)
""
#把training data切成多個(40,100,1)的小段
#似乎train不太起來
""

def preprocess():
    with open("data/mfcc.txt", "rb") as fp:
        data = pickle.load(fp)
    label=np.load("data/label.npy")
    #label=to_categorical(label,num_classes=41)

    clippeddata=np.zeros((32520,40,100,1))
    clippedlabel=np.zeros((32520,1))

    i=0
    for im in range(len(data)):
        print(im)
        cnt=0
        while (cnt+100)<len(data[im][0]):
            clippeddata[i]=data[im][:,cnt:cnt+100].reshape((40,100,1))
            clippedlabel[i]=label[im]
            cnt+=100
            i+=1
        clippeddata[i][:,:len(data[im][0])-cnt] = data[im][:,cnt:].reshape((40,len(data[im][0])-cnt,1))
        clippedlabel[i] = label[im]
        cnt += 100
        i += 1

    np.save("clippeddata.npy",clippeddata)
    np.save("clippedlabel.npy",clippedlabel)



""
#算出training data裡所有數值的平均和標準差
#把[mean,var]存成para.npy
#用於normalize
""

def normlize():
    with open("data/mfcc.txt", "rb") as fp:
        data = pickle.load(fp)

    paddeddata=np.zeros((40,2774178))
    l=0

    for i in range(len(data)):
        paddeddata[:,int(l):int(l+len(data[i][0]))]=data[i]
        l+=len(data[i][0])

    avg=np.average(paddeddata)
    var=np.sqrt((paddeddata-avg)*(paddeddata-avg))
    var=np.average(var)

    print(avg,var)
    np.save("data/para.npy",np.array([avg,var]))





""
#把每筆normalize後的data左右補0補成(40,1300)
#存成paddaddata.npy
""

def padded_data(path):
    para=np.load("data/para.npy")

    with open(path, "rb") as fp:
        data = pickle.load(fp)

    paddeddata=np.zeros((9400,40,1300))

    for i in range(len(data)):
        l=len(data[i][0])
        data[i] -= para[0]
        data[i] /= para[1]
        paddeddata[i][:,int(650-(l-l%2)/2):int(650+(l+l%2)/2)]=data[i]

    paddeddata.reshape((9400,40,1300,1))
    if path == 'data/mfcc.txt':
        np.save("data/paddeddata.npy",paddeddata)
    else:
        np.save("data/test_paddeddata.npy", paddeddata)
    #print(paddeddata,paddeddata.shape)



def train():
    label=np.load("data/label.npy")
    para=np.load("data/para.npy")
    data=np.load("data/paddeddata.npy").reshape((9473,40,1300,1))
    label=to_categorical(label,num_classes=41)

    model=Sequential()
    model.add(Conv2D(30,(3,5),input_shape=(40,None,1),border_mode="same")) #predict時input大小可以不固定
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(75,(3,5),border_mode="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(75,(3,5),border_mode="same"))
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling2D())#因為input大小不固定，所以Conv2D完要接這層後才能接Dense
    
    model.add(Flatten())

    model.add(Dense(units=500))
    model.add(Activation("relu"))
    model.add(Dense(units=250))
    model.add(Activation("relu"))
    model.add(Dense(units=41))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    if not os.path.isdir('model'):
        os.makedirs('model')
    filepath = 'model/weights-{epoch:02d}-{val_loss:.2f}.h5'
    hist = History()
    checkpoint1= ModelCheckpoint(filepath,monitor='val_acc', verbose=1,save_best_only=True, mode='max')
    checkpoint2=EarlyStopping(monitor='val_acc',min_delta=0,patience=10,verbose=0, mode='max')
    callbacks_list = [checkpoint1,checkpoint2]
    model.fit(data,label,batch_size=64,epochs=50,validation_split=0.1, shuffle=True, callbacks=callbacks_list)
    #datagen=image.ImageDataGenerator()
    #(train_generator) = datagen.flow(data[:9000], label[:9000] ,batch_size=50 )
    #model.fit_generator(train_generator, steps_per_epoch=100  ,epochs=50
    #                    ,validation_data=(data[9000:], label[9000:]), callbacks=callbacks_list)

    model.save("model/model.h5")
    
def predict(dic):
    model = load_model('model/weights-33-1.74.h5')
    testdata = np.load('data/test_paddeddata.npy')
    #label_true = np.load('data/label.npy')
    pred = model.predict(testdata.reshape((len(testdata),40,len(testdata[0][0]),1)))
    print(pred[0])
    pred = np.argsort(-pred, axis=1)
    pred_max = pred[:,0]
    print(pred[0])
    label = pred[:,:3]
    fname = os.listdir('audio_test')
    reverse_dic = dict(zip(dic.values(), dic.keys()))
    with open('result.csv', 'w') as f:
        f.write('fname,label')
        for i in range(len(label)):
            f.write('\n' + fname[i] + ',' + reverse_dic[label[i,0]]+ ' ' +
                    reverse_dic[label[i,1]]+ ' ' + reverse_dic[label[i,2]])
if __name__ == '__main__':
    main()
