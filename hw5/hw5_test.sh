#!/usr/bin/env bash
wget https://www.dropbox.com/s/kzxbld6rj2358gi/0602_gensim_0.82693.h5?dl=1
wget https://www.dropbox.com/s/8h9tcm5rvhxbn3z/gensim_w2v_0.82693_0602_model.syn1neg.npy?dl=1
wget https://www.dropbox.com/s/f13co9m0ncsgq5l/gensim_w2v_0.82693_0602_model.wv.syn0.npy?dl=1
mv 0602_gensim_0.82693.h5?dl=1 0602_gensim_0.82693.h5
mv gensim_w2v_0.82693_0602_model.syn1neg.npy?dl=1 gensim_w2v_0.82693_0602_model.syn1neg.npy
mv gensim_w2v_0.82693_0602_model.wv.syn0.npy?dl=1 gensim_w2v_0.82693_0602_model.wv.syn0.npy
python3 hw5_test.py $1 $2 $3
rm 0602_gensim_0.82693.h5
rm gensim_w2v_0.82693_0602_model.syn1neg.npy
rm gensim_w2v_0.82693_0602_model.wv.syn0.npy
