import numpy as np
import scipy.io as sio
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from scipy.signal import ellip,ellipord,sosfilt

def load_data(data_path, subject, training, all_trials = True):
	n_channels = 22
	n_tests = 6*48 	
	window_Length = 7*250

	class_return = np.zeros(n_tests)
	data_return = np.zeros((n_tests, n_channels, window_Length))

	NO_valid_trial = 0
	if training:
		a = sio.loadmat(data_path+'A0'+str(subject)+'T.mat')
	else:
		a = sio.loadmat(data_path+'A0'+str(subject)+'E.mat')
	a_data = a['data']
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2= [a_data1[0,0]]
		a_data3= a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_artifacts = a_data3[5]

		for trial in range(0,a_trial.size):
 			if(a_artifacts[trial] != 0 and not all_trials):
 			    continue
 			data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
 			class_return[NO_valid_trial] = int(a_y[trial])
 			NO_valid_trial +=1


	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]

#%%
def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

def BCI_Filter(EEGSignal): #EEGSignal [None,1,Channel,timepoints]
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
        sig_filtered[i,0,:,:] = sosfilt(sos,EEGSignal[i,0,:,:])
    return sig_filtered

def get_data_IIa(path, subject,  isFilter = False, isStandard = True):
    # Define dataset parameters
    fs = 250          # sampling rate
    t1 = int(1.5*fs)  # start time_point
    t2 = int(6*fs)    # end time_point
    T = t2-t1         # length of the MI trial (samples or time_points)

    path = path + 's{:}/'.format(subject+1)
    X_train, y_train = load_data(path, subject+1, True)
    #X_train = X_train[np.where(y_train<=2),:,:][0,:,:,:]#two classfication
    #selectchannels = [8, 9, 10, 11, 12, 15, 16, 17]#few channels
    #X_train = X_train[:,selectchannels,:]
    #y_train = y_train[np.where(y_train<=2)]
    X_test, y_test = load_data(path, subject+1, False)
    #X_test = X_test[np.where(y_test <= 2), :, :][0,:,:,:]
    #X_test = X_test[:,selectchannels,:]
    #y_test = y_test[np.where(y_test <= 2)]

    N_tr, N_ch, _ = X_train.shape 
    X_train = X_train[:, :, t1:t2].reshape(N_tr, 1, N_ch, T)
    y_train_onehot = (y_train-1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)
    N_test, N_ch, _ = X_test.shape 
    X_test = X_test[:, :, t1:t2].reshape(N_test, 1, N_ch, T)
    y_test_onehot = (y_test-1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)	

    if (isFilter ==True):
        X_train = BCI_Filter(X_train)
        X_test = BCI_Filter(X_test)

    if (isStandard == True):
        X_train, X_test = standardize_data(X_train, X_test, N_ch)
    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot