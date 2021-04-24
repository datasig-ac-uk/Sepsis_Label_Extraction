import os
import sys

import numpy as np
import torch
from torch import nn, optim

sys.path.insert(0, '../../')
import constants
from data.dataset import TimeSeriesDataset
from models.nets import LSTM
import models.LSTM.LSTM_functions as lstm_functions
import omni.functions as omni_functions
import features.mimic3_function as mimic3_myfunc


def train_LSTM(T_list, x_y, definitions, data_folder='blood_only_data/', fake_test=False):
    """

    :param T_list: (list of int) list of parameter T
    :param x_y: (list of int)list of sensitivity parameter x and y
    :param definitions:  (list of str) list of definitions. e.g.['t_suspision','t_sofa','t_sepsis_min']
    :param data_folder:   (str) folder name specifying
    :return:

    """
    for x, y in x_y:
        data_folder = 'fake_test1/' + data_folder if fake_test else data_folder
        Root_Data, Model_Dir, _, _ = mimic3_myfunc.folders(data_folder, model='LSTM')
        config_dir = constants.MODELS_DIR + 'blood_only/LSTM/hyperparameter/config'

        #     Data_Dir = Root_Data + '/processed/experiments_' + str(x) + '_' + str(y) + '/H3_subset/'
        Data_Dir = Root_Data  + 'train' + '/'

        for definition in definitions:
            config = omni_functions.load_pickle(config_dir + definition[1:])
            print(config)

            #         for T in [6]:
            for T in T_list:
                print('load timeseries dataset')
                dataset = TimeSeriesDataset().load(Data_Dir + str(x) + '_' + str(y) + definition[1:] + '_ffill.tsd')

                print('load train labels')
                labels_train = np.load(Data_Dir + 'label'+'_'+str(x)+'_'+str(y)+'_'+str(T) + definition[1:] + '.npy')

                # get torch dataloader for lstm
                train_dl, scaler = lstm_functions.prepared_data_train(dataset, labels_train, True, 128, device)

                omni_functions.save_pickle(scaler, Model_Dir + 'hyperparameter/scaler' + definition[1:])

                # specify lstm model architecture

                model = LSTM(in_channels=dataset.data.shape[-1], num_layers=1,
                             hidden_channels=config['hidden_channels'],
                             hidden_1=config['linear_channels'], out_channels=2,
                             dropout=0).to(device)

                lstm_functions.train_model(model, train_dl, n_epochs=config['epochs'],
                                           save_dir=Model_Dir + '_' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:],
                                           loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=config['lr']))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    T_list = constants.T_list[1:2]
    print(T_list)
    data_folder = constants.exclusion_rules[0]
    x_y = constants.xy_pairs[:1]

    train_LSTM(T_list,x_y,constants.FEATURES,data_folder,fake_test=False)
    """
    x_y = [(24, 12)]
    data_folder_list = constants.exclusion_rules[1:]
    for data_folder in data_folder_list:
        train_LSTM(T_list, x_y, constants.FEATURES, data_folder, fake_test=False)
    """
