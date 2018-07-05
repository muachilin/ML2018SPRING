In this directory:

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
> sh downloadModel.sh
> python3 test_spec.py testname.pkl [path/to/sample_submission.csv] [result csvfile]
