import numpy as np
import os
import sys
import torch

sys.path.insert(0, '../../../')
from definitions import *
from src.models.LSTM.lstm_functions import *
from src.features.sepsis_mimic3_myfunction import *


<<<<<<< Updated upstream
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
=======
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
        Root_Data, Model_Dir, _, _, _ = folders(data_folder, model='LSTM')
        config_dir = MODELS_DIR + 'blood_only_data/LSTM/hyperparameter/config'
>>>>>>> Stashed changes

        Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/train/'

        for definition in definitions:
<<<<<<< Updated upstream
            config = load_pickle(MODELS_DIR + 'hyperparameter/LSTM/config' + definition[1:])
=======
            config = load_pickle(config_dir + definition[1:])
            print(config)

            #         for T in [6]:
>>>>>>> Stashed changes
            for T in T_list:
                print('load timeseries dataset')
                dataset = TimeSeriesDataset().load(Data_Dir + definition[1:] + '_ffill.tsd')

                print('load train labels')
                labels_train = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

                # get torch dataloader for lstm
                train_dl, scaler = prepared_data_train(dataset, labels_train, True, 128, device)

                save_pickle(scaler, Model_Dir + 'hyperparameter/scaler' + definition[1:])
<<<<<<< Updated upstream
                # specify lstm model architecture with tuned hyperparameter s
=======

                # specify lstm model architecture
>>>>>>> Stashed changes

                model = LSTM(in_channels=dataset.data.shape[-1], num_layers=1,
                             hidden_channels=config['hidden_channels'],
                             hidden_1=config['linear_channels'], out_channels=2,
                             dropout=0).to(device)

                train_model(model, train_dl, n_epochs=config['epochs'],
                            save_dir=Model_Dir + '_' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:],
                            loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=config['lr']))
<<<<<<< Updated upstream


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    data_folder = 'blood_only_data/'
    train_LSTM(T_list,xy_pairs,definitions,data_folder)
=======


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    xy_pairs=[(48,24),(12,6),(6,3)]
    definitions = ['t_sofa','t_suspicion','t_sepsis_min']
    #train_LSTM(T_list, xy_pairs, definitions, data_folder='blood_only_data/', fake_test=False)
    xy_pairs = [(24,12)]
    data_folder_list = ['absolute_values/','strict_exclusion/','all_cultures/','no_gcs/']
    for data_folder in data_folder_list:
        train_LSTM(T_list, xy_pairs, definitions, data_folder=data_folder, fake_test=False)
>>>>>>> Stashed changes
