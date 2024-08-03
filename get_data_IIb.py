import numpy as np
import scipy.io as sio
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from scipy.signal import ellip,ellipord,sosfilt

def load_data(data_path,subject, train):
    windowlength = int(7.5*250)
    if train == True:
        a = sio.loadmat(data_path+'B0'+str(subject+1)+'T.mat')
    else:
        a = sio.loadmat(data_path+'B0'+str(subject+1)+'E.mat')
    a_data = a["data"]
    list_a = []
    list_L = []
    for ii in range(0,a_data.shape[1]):
        origin_data = a_data[0,ii][0][0][0]
        trial_begin = a_data[0,ii][0][0][1]
        label = a_data[0,ii][0][0][2][:,0]
        c = [origin_data[i:i + windowlength] for i in trial_begin[:,0].astype(int)]
        if ii == 0:
            list_a = c
            list_L = label
        else:
            list_a.extend(c)
            list_L = np.concatenate((list_L,label))
    X = np.array(list_a)[:,:,:3]
    y = np.array(list_L)
    return X,y

def standardize_data(X_train, X_test, channels):
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

def BCI_Filter(EEGSignal):
    Fs = 250
    Wp = [6/(Fs/2),28/(Fs/2)]
    Ws = [5.5/(Fs/2),28.5/(Fs/2)]
    Rp = 0.5
    Rs = 25
    N,Wn = ellipord(Wp,Ws,Rp,Rs,analog=False,fs=None)
    sos = ellip(N,Rp,Rs,Wn,btype='bandpass', analog=False, output='sos', fs=None)
    N_tr,ch,N_ch,T = EEGSignal.shape
    sig_filtered = np.zeros((N_tr,ch,N_ch,T))
    for i in range(N_tr):
        sig_filtered[i,0,:,:] = sosfilt(sos,EEGSignal[i,0:,:])
    return sig_filtered

def get_data_IIb(data_path, subject, isFilter = False,isStandard=True):
    fs = 250
    t1 = int(2 * fs)
    t2 = int(7.5 * fs)
    T = t2 - t1
    X_train, y_train = load_data(data_path, subject, train=True)
    X_train = X_train.swapaxes(1,2)
    X_test, y_test = load_data(data_path, subject, train=False)
    X_test = X_test.swapaxes(1,2)

    N_tr,  N_ch,_ = X_train.shape
    X_train = X_train[:, :, t1:t2].reshape(N_tr, 1, N_ch, T)
    y_train_onehot = (y_train - 1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)

    N_test,  N_ch,_ = X_test.shape
    X_test = X_test[:, :, t1:t2].reshape(N_test, 1, N_ch, T)
    y_test_onehot = (y_test - 1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)
    if (isStandard == True):
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    if (isFilter == True):
        X_train = BCI_Filter(X_train)
        X_test = BCI_Filter(X_test)
    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


