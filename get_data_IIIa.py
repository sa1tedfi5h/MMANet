import numpy as np
from keras.utils import to_categorical
import scipy.io as sio
from get_data_IIa import standardize_data
# load data from 05_IIIa

def get_data_IIIa(data_path,subject,isshuffle=True,isStandard = True):
    data_load = sio.loadmat(data_path + 'data_IIIa_subj0' + str(subject+1))
    data = data_load['data']
    label = data_load['labels']
    label = label[:, 0]
    data = data.swapaxes(2, 0)
    data_shape = data.shape
    N_tr = data_shape[0]  # number of trials
    N_chan = data_shape[1]  # number of channels
    N_s = data_shape[2]  # number of sample points
    # split data for training set and testing set, half data for training,half data for testing
    data_1 = data[np.where(label == 1),:,:][0,:,:,:]  # find data for class 1
    data_2 = data[np.where(label == 2),:,:][0,:,:,:]  # find data for class 2
    data_3 = data[np.where(label == 3),:,:][0,:,:,:] # find data for class 3
    data_4 = data[np.where(label == 4),:,:][0,:,:,:]  # find data for class 4
    data_11 = data_1[:int(N_tr / 8), :, :]
    data_21 = data_2[:int(N_tr / 8), :, :]
    data_31 = data_3[:int(N_tr / 8), :, :]
    data_41 = data_4[:int(N_tr / 8), :, :]
    data_12 = data_1[int(N_tr / 8):, :, :]
    data_22 = data_2[int(N_tr / 8):, :, :]
    data_32 = data_3[int(N_tr / 8):, :, :]
    data_42 = data_4[int(N_tr / 8):, :, :]
    x_train = np.concatenate((data_11,data_21,data_31,data_41),axis=0)
    x_test = np.concatenate((data_12,data_22,data_32,data_42),axis=0)
    y_train_1 = np.ones(int(N_tr / 8))
    y_train_2 = 2 * y_train_1
    y_train_3 = 3*y_train_1
    y_train_4 = 4*y_train_1
    y_train = np.concatenate((y_train_1, y_train_2,y_train_3,y_train_4), axis=0)
    y_test = y_train
    if isshuffle==True:
        indics = np.arange(int(N_tr / 2))
        np.random.seed(117)
        np.random.shuffle(indics)
        x_train = x_train[indics,:,:]
        y_train = y_train[indics]
        x_test = x_test[indics,:,:]
        y_test = y_test[indics]
    y_train_onehot = (y_train - 1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)
    y_test_onehot = y_train_onehot
    x_train = x_train.reshape(int(N_tr / 2), 1, N_chan, N_s)
    x_test = x_test.reshape(int(N_tr / 2), 1, N_chan, N_s)
    if isStandard == True:
        N_ch = x_train.shape[2]
        x_train, x_test = standardize_data(x_train, x_test, N_ch)
    return x_train,y_train,y_train_onehot,x_test,y_test,y_test_onehot


