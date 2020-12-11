import os
import sys

import numpy as np
import torch
from torch import nn, optim

sys.path.insert(0, '../../')
import constants
from data.dataset import TimeSeriesDataset
from models.nets import LSTM
import models.LSTM.lstm_functions as lstm_functions
import omni.functions as omni_functions
import features.sepsis_mimic3_myfunction as mimic3_myfunc


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    T_list=[6]
    for x, y in [ (24, 12)]:

        current_data_folder = 'absolute_values/'
        Root_Data, Model_Dir, Data_save = mimic3_myfunc.folders(current_data_folder,model='LSTM')

        #     Data_Dir = Root_Data + '/processed/experiments_' + str(x) + '_' + str(y) + '/H3_subset/'
        Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/train/'

        for definition in constants.FEATURES:
            config = omni_functions.load_pickle(constants.MODELS_DIR + 'hyperparameter/LSTM/config' + definition[1:])
        #         for T in [6]:
            for T in T_list:
                print('load timeseries dataset')
                dataset = TimeSeriesDataset().load(Data_Dir + definition[1:] + '_ffill.tsd')

                print('load train labels')
                labels_train = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

                # get torch dataloader for lstm
                train_dl, scaler = lstm_functions.prepared_data_train(dataset, labels_train, True, 128, device)

                omni_functions.save_pickle(scaler, Model_Dir + 'hyperparameter/scaler' + definition[1:])
                #             save_pickle(scaler,MODELS_DIR+'/LSTM/hyperparameter/scaler'+definition[1:]+'H3_subset')

                # specify lstm model architecture

                model = LSTM(in_channels=dataset.data.shape[-1], num_layers=1, hidden_channels=config['hidden_channels'],
                         hidden_1=config['linear_channels'], out_channels=2,
                         dropout=0).to(device)

                lstm_functions.train_model(model, train_dl, n_epochs=config['epochs'],
                        save_dir=Model_Dir + '_' + str(x) + '_' + str(y) + '_' + str(T)+ definition[1:],
                        loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=config['lr']))

#             train_model(model, train_dl, n_epochs=config['epochs'],
#                     save_dir=MODELS_DIR + '/LSTM/H3_subset/' +definition[2:] +'_'+ str(x) + '_' + str(y) + '_' + str(T),
#                     loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=config['lr']))
