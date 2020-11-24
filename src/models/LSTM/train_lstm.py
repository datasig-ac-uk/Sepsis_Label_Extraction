import numpy as np
import os
import sys
import torch

sys.path.insert(0, '../../../')
from definitions import *
from src.models.LSTM.lstm_functions import *
from src.features.sepsis_mimic3_myfunction import *


def train_LSTM(T_list, x_y, definitions, data_folder):
    """
    Training on the CoxPHM model for specified T,x_y and definition parameters, save the trained model in model
    directory

    :param T_list(list of int): list of parameter T
    :param x_y(list of int):list of sensitivity parameter x and y
    :param definitions(list of str): list of definitions. e.g.['t_suspision','t_sofa','t_sepsis_min']
    :param data_folder(str): folder name specifying
    :return: save trained model
    """
    for x, y in x_y:

        Root_Data, Model_Dir, Data_save, _, _ = folders(data_folder, model='LSTM')

        Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/train/'

        for definition in definitions:
            config = load_pickle(MODELS_DIR + 'hyperparameter/LSTM/config' + definition[1:])
            for T in T_list:
                print('load timeseries dataset')
                dataset = TimeSeriesDataset().load(Data_Dir + definition[1:] + '_ffill.tsd')

                print('load train labels')
                labels_train = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

                # get torch dataloader for lstm
                train_dl, scaler = prepared_data_train(dataset, labels_train, True, 128, device)

                save_pickle(scaler, Model_Dir + 'hyperparameter/scaler' + definition[1:])
                # specify lstm model architecture with tuned hyperparameter s

                model = LSTM(in_channels=dataset.data.shape[-1], num_layers=1,
                             hidden_channels=config['hidden_channels'],
                             hidden_1=config['linear_channels'], out_channels=2,
                             dropout=0).to(device)

                train_model(model, train_dl, n_epochs=config['epochs'],
                            save_dir=Model_Dir + '_' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:],
                            loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=config['lr']))


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    data_folder = 'blood_only_data/'
    train_LSTM(T_list,xy_pairs,definitions,data_folder)