from setting import *
from ext import *
import numpy as np
import librosa

def data_load():
    """ loading MFCC's std & mean"""
    import pandas as pd
    import numpy as np
    import os, datetime
    import pickle
    if os.path.isfile('data/mfcc.pkl'):
        feats = pickle.load(open("data/mfcc.pkl",'rb'))
        labels = pickle.load(open("data/baseline_labels.pkl",'rb'))
    else:
        y = get_df_fma()
        ids = np.array(y.keys())
        feats, labels = [], []
        for file in os.listdir('D:/Download/dataset/fma_small/'):
            for f in os.listdir('D:/Download/dataset/fma_small/'+file):
                track_id = int(os.path.splitext(f)[0])
                if track_id in ids:
                    try:
                        t0 = datetime.datetime.now()
                        signal, sr = librosa.load('D:/Download/dataset/fma_small/'+file+'/'+f)
                        mfcc = librosa.feature.mfcc(signal, sr, n_mfcc=13)
                        mfcc = np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1)), axis=0)
                        feats.append(mfcc)
                        labels.append(y.loc[track_id])
                        print (datetime.datetime.now()-t0)
                    except:
                        print ('#####D:/Download/dataset/fma_small/'+file+'/'+f)
        pickle.dump(feats, open("data/mfcc.pkl",'wb'))
        pickle.dump(labels, open("data/baseline_labels.pkl",'wb'))
    return np.array(feats), np.array(labels)

def feats_load():
    """ loading features from pre-trained convnet """
    import pickle, os
    import pandas as pd
    if os.path.isfile('data/feats_n.pkl'):
        feats = pickle.load(open("data/feats_n.pkl",'rb'))
        labels = pickle.load(open("data/labels_n.pkl",'rb'))
    else:
        feat_extractor = model_load()
        y = get_df_fma()
        ids = np.array(y.keys())
        feats, labels = [], []
        for file in os.listdir('D:/Download/dataset/fma_small/'):
            for f in os.listdir('D:/Download/dataset/fma_small/'+file):
                track_id = int(os.path.splitext(f)[0])
                if track_id in ids:
                    try:
                        t0 = datetime.datetime.now()
                        signal, sr = librosa.load('D:/Download/dataset/fma_small/'+file+'/'+f, sr=12000)
                        feat = feat_extractor.predict(signal[np.newaxis, np.newaxis])
                        feats.append(feat)
                        labels.append(y.loc[track_id])
                        print (datetime.datetime.now()-t0)
                    except:
                        print ('#####D:/Download/dataset/fma_small/'+file+'/'+f)
        pickle.dump(feats, open("data/feats_n.pkl",'wb'))
        pickle.dump(labels, open("data/labels_n.pkl",'wb'))
    return feats, labels

def log_scale_melspectrogram(path):
    """ extracting mel-spectrogram wrapper """
    signal, sr = librosa.load(path, sr=Fs)
    n_sample = signal.shape[0]
    n_sample_fit = int(DURA*Fs)

    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
    elif n_sample > n_sample_fit:
        signal = signal[(n_sample-n_sample_fit)//2:(n_sample+n_sample_fit)//2]

    melspect = librosa.logamplitude(librosa.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2, ref_power=1.0)
    return melspect

def one_hot_encoder(labels):
    """ customize one hot encoder wrapper """
    num_labels = labels.shape[0]
    labels_1h = np.zeros((num_labels, len(genres)))
    for i in range(0, num_labels):
        labels_1h[i] = ((labels[i]==np.array(genres)).astype(int))
    return labels_1h

def get_df_fma():
    """ fma labels wrapper """
    import pandas as pd
    tracks = pd.read_csv('fma/tracks.csv', index_col=0, header=[0,1])
    features = pd.read_csv('fma/features.csv', index_col=0, header=[0,1,2])
    small = tracks['set', 'subset'] == 'small'
    y = tracks.loc[small, ('track', 'genre_top')]
    return y

def cross_validation(x, y, cv=5, seed=1):
    """ Cross Validation wrapper """
    """ x: data
        y: labels
        cv: k-fold
        seed: random seed
    """
    from sklearn.model_selection import StratifiedKFold
    sf = StratifiedKFold(n_splits=cv, random_state=seed, shuffle=False)
    for train_index, test_index in sf.split(x, y):
        yield train_index, test_index

def z_score_normalization(train, test):
    """ Z score Normalization """
    """ Formula: (x - mean)/std """
    import sklearn
    scaler = sklearn.preprocessing.StandardScaler(copy=False)
    train_ = scaler.fit_transform(train)
    test_ = scaler.transform(test)
    return train_, test_

def get_metrics(expected, predicted):
    """ mertics wrapper """
    from sklearn import metrics
    f1_score = metrics.f1_score(expected, predicted, average='weighted')
    recall_score = metrics.recall_score(expected, predicted, average='weighted')
    precision_score = metrics.precision_score(expected, predicted, average='weighted')
    accuracy_score = metrics.accuracy_score(expected, predicted)
    return [accuracy_score, recall_score, precision_score, f1_score]