# final
## data structure:

- /data
  - mfcc.txt
  - test_mfcc.txt
  - train.csv
  - sample_submission.csv
  - label.npy
  - paddeddata.npy
  - test_paddeddata.npy
  - para.npy
  - train_X_verified.npy
  - train_Y_verified.npy
  - train_X_nonveri.npy
  - train_Y_nonveri.npy
- /model
- /audio_test
- /audio_train
- preprocess_test.py
- preprocess_train.py
- train_mfcc.py
- train_spec.py
- result.csv

## usage:
In the directory : src/spectrosegment/

1. Convert wav files into spectrograms. You may set the variable 'VERIFIED' in preprocess_train.py to convert verified training data or non-verified training data.
> python3 preprocess_train.py [path/to/train.csv] [path/to/audio_train]

output:
	train_X_nonveri.npy
	train_Y_nonveri.npy
	train_X_verified.npy
	train_Y_verified.npy

> python3 preprocess_test.py path/to/audio_test testname

output:
	testname.pkl 

2. Train CNN model by verified data.
> python3 train_spec.py [modelname.h5] train

output:
	modelname.h5

3. Filter out non-verified data that are unreliable.
> python3 preprocess_nonveri.py [path/to/modelname.h5] train_X_nonveri.npy train_Y_nonveri.npy

output:
	X_nonveri_filtered.npy
	Y_nonveri_filtered.npy

4. Train CNN by verified data and 'reliable' non-verified data.
> python3 train_spec.py [newmodelname.h5] retrain

output:
	newmodelname.h5

5. Predict testing data.
>sh downloadModel.sh
> python3 test_spec.py testname.pkl [path/to/sample_submission.csv] [result csvfile]

## tool-kit version
1. Keras==2.0.8
2. Tensorflow==1.4.0
3. Librosa==0.6.0



