import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import keras
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score,roc_auc_score,f1_score
from model import MMANet
from keras.models import Model
from keras.utils import to_categorical
from get_data_IIa import get_data_IIa
from get_data_IIIa import get_data_IIIa
from get_data_IIb import get_data_IIb

plt.rcParams['font.family'] = 'Times New Roman'  #  Times New Roman

def draw_learning_curves(history,results_path,sub):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + '/subject_' + sub + '_acc.png')
    #plt.show()
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper right')
    plt.savefig(results_path + '/subject_' + sub + '_loss.png')
    #plt.show()
    plt.close()

def draw_confusion_matrix(cf_matrix, sub, results_path):
    # Generate confusion matrix plot
    display_labels = ['Left hand', 'Right hand','Foot','Tongue']
    # display_labels = ['left hand','right hand'] # dataset IIb
    cmap = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                display_labels=display_labels)
    disp.plot(cmap=cmap)
    disp.ax_.set_xticklabels(display_labels, rotation=0)
    plt.title('Confusion Matrix of Subject: ' + sub )
    plt.savefig(results_path + '/subject_' + sub + '.png')
    # plt.show()

def draw_performance_barChart(num_sub, metric, label, results_path,title):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label, color = 'mediumslateblue')
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject')
    ax.set_ylim([0,1])
    plt.savefig(results_path + title +  '.png')

def draw_heat_map(X,results_path,sub):

    data = pd.DataFrame(X)
    ax = sns.heatmap(data,cmap="YlGnBu", cbar_kws={"location": "right", "pad": 0.08})
    plt.xlabel('Index of Spatial Pattern',size = 12)
    plt.ylabel('EEG Channel',size = 12,rotation = 90)
    y_ticks = [ i + 0.5 for i in range(22)]
    y_ticks_label = [ i + 1 for i in range(22)]
    x_ticks = [ i + 0.5  for i in range(9)]
    x_ticks_label = [ i + 1 for i in range(9)]
    plt.xticks(x_ticks,x_ticks_label)
    plt.yticks(y_ticks,y_ticks_label)
    #plt.title('HeatMap_W',size = 20)

    ax2 = ax.twinx()
    y2_ticks_label = ['Fz','Fc3','Fc1','Fcz','Fc2','Fc4','C5','C3','C1','Cz','C2','C4','C6','Cp3','Cp1','Cpz','Cp2','Cp4','P1','Pz','P2','Poz']
    y2_ticks_label.reverse()
    ax2.set_yticks([ i + 0.5 for i in range(22)])
    ax2.set_yticklabels(y2_ticks_label)
    ax2.set_ylim(0, 22)
    plt.savefig(results_path + '/subject_' + sub + '_HeatMap.jpg',dpi=1200)
    plt.show()

def draw_TSNE_Visualization(X,treu_label,results_path,sub):
    #label_names = ["left hand", "right hand", "foot", "tongue"]
    unique_labels = np.unique(treu_label)
    #colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    colors = ['red','blue','green','purple']
    for i, label in enumerate(unique_labels):
        subset = X[treu_label == label]
        plt.scatter(subset[:, 0], subset[:, 1], color=colors[int(i)]) #label=label_names[int(label-1)])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.xticks(np.arange(min(X[:, 0]), max(X[:, 0]), step=4.0))
    plt.yticks(np.arange(min(X[:, 1]), max(X[:, 1]), step=4.0))
    #plt.legend(loc="upper left")
    #plt.title('Visualization', size=20)
    plt.savefig(results_path + '/subject_' + sub + '_TSNE.jpg',dpi=1200)
    plt.show()

#%% Training
def train(Dataset_config, Train_config, results_path):
    in_exp = time.time()
    best_models = open(results_path + "/best models.txt", "w")
    log_write = open(results_path + "/log.txt", "w")
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')
    
    # dataset paramters
    n_sub = Dataset_config.get('n_sub')
    data_path = Dataset_config.get('data_path')
    isStandard = Dataset_config.get('isStandard')
    isFilter = Dataset_config.get('isFilter')
    #training hyperparamters
    batch_size = Train_config.get('batch_size')
    epochs = Train_config.get('epochs')
    patience = Train_config.get('patience')
    lr = Train_config.get('lr')
    LearnCurves = Train_config.get('LearnCurves') # Plot Learning Curves?
    n_train = Train_config.get('n_train')
    model_name = Train_config.get('model')

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))
    AUC = np.zeros((n_sub,n_train))
    F1_score = np.zeros((n_sub,n_train))

    for sub in range(n_sub):
        #  calculate the subject training time
        in_sub = time.time()
        print('\nTraining on subject ', sub+1)
        log_write.write( '\nTraining on subject '+ str(sub+1) +'\n')
        # Initiating variables to save the best subject accuracy among multiple runs.
        BestSubjAcc = 0 
        bestTrainingHistory = [] 
        # Get training and test data
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data_IIa(data_path, sub,isFilter, isStandard)
        # X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data_IIIa(data_path,sub,isshuffle=True,isStandard = True)
        # X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data_IIb(data_path,sub,isFilter,isStandard)

        for train in range(n_train): # How many repetitions of training for subject i.
            #  calculate the 'run' training time
            in_run = time.time()
            filepath = results_path + '/saved models/run-{}'.format(train+1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)        
            filepath = filepath + '/subject-{}.h5'.format(sub+1)
            
            # Create the model
            model = getModel(model_name)
            # Compile and train the model
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])          
            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, 
                                save_best_only=True, save_weights_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]
            history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), 
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
            # keras.utils.plot_model(model, to_file="my_network.png", show_shapes=True)

            model.load_weights(filepath)
            y_pred_onehot = model.predict(X_test)
            y_pred = y_pred_onehot.argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train]  = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)
            AUC[sub, train] = roc_auc_score(labels, y_pred_onehot,multi_class='ovr') # y_pred
            # AUC[sub, train] = roc_auc_score(labels, y_pred, multi_class='ovr') # IIb y_pred
            F1_score[sub,train] = f1_score(labels,y_pred,average='weighted')
            # Get the current 'OUT' time to calculate the 'run' training time
            out_run = time.time()
            # Print & write performance measures for each run
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub+1, train+1, ((out_run-in_run)/60))
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}  Test_AUC:{:.4f}  Test_F1-score:{:.4f}'.format(acc[sub, train], kappa[sub, train],AUC[sub,train],F1_score[sub,train])
            print(info)
            log_write.write(info +'\n')
            # If current training run is better than previous runs, save the history.
            if(BestSubjAcc < acc[sub, train]):
                 BestSubjAcc = acc[sub, train]
                 bestTrainingHistory = history
        
        # Store the path of the best model among several runs
        best_run = np.argmax(acc[sub,:])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run+1, sub+1)+'\n'
        best_models.write(filepath)
        # Get the current 'OUT' time to calculate the subject training time
        out_sub = time.time()
        # Print & write the best subject performance among multiple runs
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub+1, best_run+1, ((out_sub-in_sub)/60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]), acc[sub,:].std() )
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}   '.format(kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub,:].std())
        info = info + 'AUC: {:.4f}   avg_AUC: {:.4f} +- {:.4f}   '.format(AUC[sub, best_run], np.average(AUC[sub, :]),AUC[sub, :].std())
        info = info + 'F1-score: {:.4f}   avg_F1-score: {:.4f} +- {:.4f}'.format(F1_score[sub, best_run], np.average(F1_score[sub, :]),
                                                                       F1_score[sub, :].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info+'\n')
        # Plot Learning curves 
        if (LearnCurves == True):
            print('Plot Learning Curves ....... ')
            draw_learning_curves(bestTrainingHistory,results_path,str(sub+1))
          
    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format( (out_exp-in_exp)/(60*60) )
    print(info)
    log_write.write(info+'\n')
    # accuracy/kappa over all runs.
    np.savez(perf_allRuns, acc = acc, kappa = kappa, AUC = AUC,F1_score=F1_score)

    best_models.close()
    log_write.close()
    perf_allRuns.close()

def test(model, Dataset_config, results_path, allRuns = True):
    # Open the  "Log" file to write the evaluation results 
    log_write = open(results_path + "/log.txt", "a")
    # Open the file that stores the path of the best models
    best_models = open(results_path + "/best models.txt", "r")
    n_classes =  Dataset_config.get('n_classes')
    n_sub =  Dataset_config.get('n_sub')
    data_path =  Dataset_config.get('data_path')
    isStandard =  Dataset_config.get('isStandard')
    isFilter =  Dataset_config.get('isFilter')
    # Initialize variables
    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)
    AUC_bestRun = np.zeros(n_sub)
    F1_score_bestRun = np.zeros(n_sub)
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])
    if(allRuns):
        perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
        perf_arrays = np.load(perf_allRuns)
        acc_allRuns = perf_arrays['acc']
        kappa_allRuns = perf_arrays['kappa']
        AUC_allRuns = perf_arrays['AUC']
        F1_score_allRuns = perf_arrays['F1_score']

    for sub in range(n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Load data
        _, _, _, X_test, y_test, y_test_onehot = get_data_IIa(data_path, sub, isFilter, isStandard)
        # _, _, _, X_test, _, y_test_onehot = get_data_IIIa(data_path, sub, isshuffle=True,
        #                                                                     isStandard=True)
        # _, _, _, X_test, _, y_test_onehot = get_data_IIb(data_path, sub, isFilter,isStandard)


        # Load the best
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1])
        keras.utils.plot_model(model, to_file="my_network.png", show_shapes=True)
        # Predict
        y_pred_onehot = model.predict(X_test)
        y_pred = y_pred_onehot.argmax(axis=-1)
        # Calculate accuracy and K-score
        labels = y_test_onehot.argmax(axis=-1)
        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
        AUC_bestRun[sub] = roc_auc_score(labels, y_pred_onehot,multi_class='ovr')
        # AUC_bestRun[sub] = roc_auc_score(labels, y_pred, multi_class='ovr')  # IIb y_pred
        F1_score_bestRun[sub] = f1_score(labels,y_pred,average='weighted')
        # Calculate and draw confusion matrix
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='pred')
        draw_confusion_matrix(cf_matrix[sub, :, :], str(sub+1), results_path)

        #draw the heatmap of EGG Channel weights
        # W_heatmap = model.get_layer(name='eeg_channel_attention').get_weights()[0][:,0,:]
        # draw_heat_map(W_heatmap, results_path, str(sub+1))

        # TSNE Visualization
        # intermediate = Model(inputs = model.input, outputs = model.get_layer('lambda_1').output+
        #                                                      model.get_layer('lambda_2').output+model.get_layer('lambda_3').output+
        #                                                      model.get_layer('lambda_4').output+model.get_layer('lambda_5').output)
        # intermediate = Model(inputs=model.input, outputs=model.get_layer('lambda_1').output)
        # intermediate_output = intermediate.predict(X_test)
        # tsne = TSNE(n_components=2, random_state=0)
        # tsne_result = tsne.fit_transform(intermediate_output)
        # draw_TSNE_Visualization(tsne_result,y_test,results_path,str(sub+1))

        # Print & write performance measures for each subject
        info = 'Subject: {}   best_run: {:2}  '.format(sub+1, (filepath[filepath.find('run-')+4:filepath.find('/sub')]) )
        info = info + 'acc: {:.4f}   kappa: {:.4f}   AUC:{:.4f}   F1-score:{:.4f} '.format(acc_bestRun[sub], kappa_bestRun[sub] ,AUC_bestRun[sub],F1_score_bestRun[sub])
        if(allRuns): 
            info = info + 'avg_acc: {:.4f} +- {:.4f}   avg_kappa: {:.4f} +- {:.4f}  avg_AUC:{:.4f} +- {:.4f}  avg_F1-score:{:.4f} +- {:.4f}'.format(
                np.average(acc_allRuns[sub, :]), acc_allRuns[sub,:].std(),
                np.average(kappa_allRuns[sub, :]), kappa_allRuns[sub,:].std(),
            np.average(AUC_allRuns[sub,:]),AUC_allRuns[sub,:].std(),
            np.average(F1_score_allRuns[sub,:]),F1_score_allRuns[sub,:].std())
        print(info)
        log_write.write('\n'+info)
      
    # Print & write the average performance measures for all subjects     
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}   AUC = {:.4f}  F1-score = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun), np.average(AUC_bestRun),np.average(F1_score_bestRun))
    if(allRuns): 
        info = info + '\nAverage of {} subjects x {} runs (average of {} experiments):\nAccuracy = {:.4f}   Kappa = {:.4f}   AUC = {:.4f}  F1-score = {:.4f}'.format(
            n_sub, acc_allRuns.shape[1], (n_sub * acc_allRuns.shape[1]),
            np.average(acc_allRuns), np.average(kappa_allRuns),np.average(AUC_allRuns),np.average(F1_score_allRuns))
    print(info)
    log_write.write(info)
    
    # Draw a performance bar chart for all subjects 
    draw_performance_barChart(n_sub, acc_bestRun, 'Accuracy', results_path,title = 'Accuracy')
    draw_performance_barChart(n_sub, kappa_bestRun, 'K-score',results_path,title = 'K-score')
    # Draw confusion matrix for all subjects (average)
    draw_confusion_matrix(cf_matrix.mean(0), 'All', results_path)
    # Close open files     
    log_write.close() 

def getModel(model_name):
    # Select the model
    model = MMANet(
        # Dataset parameters
        n_classes=4, # IIa /IIIa 4   dataset IIb 2
        in_chans=22, # IIb 3  IIa 22  IIIa  60
        in_samples=1125, # IIb 1375 IIa 1125 IIIa 1000
        # Sequential Slice parameter
        n_Slice=5, #IIb:3 IIa:5 IIIa:5
        diff=1, # IIb:4  IIb:1  IIIa:1
        # EEGIA parameters
        eegn_F1=16,
        eegn_D=2,
        eegn_kernelSize=64,
        eegn_poolSize=7, # dataset IIa or IIb:7 ; IIIa :6
        eegn_dropout=0.3,
         # PTSA parameters
        tcn_depth=2,
        tcn_kernelSize=4,
        tcn_filters=32,
        tcn_dropout=0.3,
        tcn_activation='elu'
            )
    return model

def run():
    # Get dataset path
    #data_path = os.path.expanduser('~') + '/BCI Competition IV/BCI Competition IV-2a/'
    data_path='./BCI Competition IV-2a/'
    # data_path='./BCI Competition IV-2b/'
    # data_path = './BCI Competition III-IIIa/'

    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results_IIa" # results of IIa
    # results_path = os.getcwd() + "/results_IIb" # results of IIb
    # results_path = os.getcwd() + "/results_IIIa" # results of IIIa

    if not  os.path.exists(results_path):
      os.makedirs(results_path)   # Create a new directory
      
    # paramters for dataset IIa
    Dataset_config = { 'n_classes': 4, 'n_sub': 9, 'n_channels': 22, 'data_path': data_path,
                'isStandard': True, 'isFilter':False}

    # parameters for dataset IIb
    # Dataset_config = {'n_classes': 2, 'n_sub': 9, 'n_channels': 3, 'data_path': data_path,
    #                 'isFilter':False,'isStandard': True}

    # # parameters for dataset IIIa
    # Dataset_config = {'n_classes': 4, 'n_sub': 3, 'n_channels': 60, 'data_path': data_path,
    #                 'isStandard': True}

    # Set training hyperparamters
    Train_config = { 'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.0009,
                  'LearnCurves': True, 'n_train': 10, 'model':'ATCNet'}
    #Train the model
    #train(Dataset_config, Train_config, results_path)

    # Evaluate the model based on the weights saved in the '/results' folder
    model = getModel(Train_config.get('model'))
    test(model, Dataset_config, results_path)
    
#%%
if __name__ == "__main__":
    run()
    