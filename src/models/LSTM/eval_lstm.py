import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc
import torch

from visualization.plot_functions import suboptimal_choice_patient

sys.path.insert(0, '../../')
import constants
from data.dataset import TimeSeriesDataset
import omni.functions as omni_functions
import models.LSTM.lstm_functions as lstm_functions
from models.nets import LSTM
import features.sepsis_mimic3_myfunction as mimic3_myfunc
from visualization.sepsis_mimic3_myfunction_patientlevel import decompose_cms, output_at_metric_level


def eval_LSTM(T_list, x_y, definitions, data_folder, train_test, thresholds=np.arange(10000) / 10000, fake_test=False):
    """
    This function compute evaluation on trained CoxPHM model, will save the prediction probability and numerical results
    for both online predictions and patient level predictions in the outputs directoty.

    :param T_list :(list of int) list of parameter T
    :param x_y:(list of int)list of sensitivity parameter x and y
    :param definitions:(list of str) list of definitions. e.g.['t_suspision','t_sofa','t_sepsis_min']
    :param data_folder:(str) folder name specifying
    :param train_test:(bool)True: evaluate the model on train set, False: evaluate the model on test set
    :param thresholds:(np array) A discretized array of probability thresholds for patient level evaluation
    :return: save prediction probability of online predictions, and save the results
              (auc/specificity/accuracy) for both online predictions and patient level predictions
    """
    results = []
    results_patient_level = []
    data_folder = 'fake_test1/' + data_folder if fake_test else data_folder
    config_dir = constants.MODELS_DIR + 'blood_only_data/LSTM/hyperparameter/config'
    Root_Data, Model_Dir, Data_save, Output_predictions, Output_results = mimic3_myfunc.folders(data_folder,
                                                                                                model='LSTM')
    for x, y in x_y:

        Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/' + train_test + '/'

        #     Save_Dir = DATA_DIR + '/processed/experiments_' + str(x) + '_' + str(y) + '/H3_subset/'

        #         for T in [6]:
        for T in T_list:
            for definition in definitions:
                config = omni_functions.load_pickle(config_dir + definition[1:])
                print('load timeseries dataset')
                dataset = TimeSeriesDataset().load(Data_Dir + definition[1:] + '_ffill.tsd')

                print('load train labels')
                labels_test = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')
                # get torch dataloader for lstm
                scaler = omni_functions.load_pickle(Model_Dir + 'hyperparameter/scaler' + definition[1:])
                #               scaler=load_pickle(MODELS_DIR+'/LSTM/hyperparameter/scaler'+definition[1:]+'H3_subset')
                test_dl = lstm_functions.prepared_data_test(dataset, labels_test, True, scaler, 1000, device)

                # specify torch model architecture and load trained model
                model = LSTM(in_channels=dataset.data.shape[-1], num_layers=1,
                             hidden_channels=config['hidden_channels'],
                             hidden_1=config['linear_channels'], out_channels=2,
                             dropout=0).to(device)

                model.load_state_dict(
                    torch.load(Model_Dir + '_' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:],
                               map_location=torch.device('cpu')))

                auc_score, specificity, accuracy, true, preds = lstm_functions.eval_model(test_dl, model,
                                                                                          save_dir=Output_predictions + str(
                                                                                              x) + '_' + str(y) + '_'
                                                                                                   + str(
                                                                                              T) + definition[
                                                                                                   1:] + '_' + train_test + '.npy')
                df_sepsis = pd.read_pickle(Data_Dir + definition[1:] + '_dataframe.pkl')
                preds = np.load(Output_predictions + str(x) + '_' + str(y) + '_'
                                + str(T) + definition[1:] + '_' + train_test + '.npy')
                CMs, _, _ = suboptimal_choice_patient(df_sepsis, true, preds, a1=6, thresholds=thresholds,
                                                      sample_ids=None)
                tprs, tnrs, fnrs, pres, accs = decompose_cms(CMs)

                results_patient_level.append(
                    [str(x) + ',' + str(y), T, definition, "{:.3f}".format(auc(1 - tnrs, tprs)),
                     "{:.3f}".format(output_at_metric_level(tnrs, tprs, metric_required=[0.85])),
                     "{:.3f}".format(output_at_metric_level(accs, tprs, metric_required=[0.85]))])

                # auc_score, specificity, accuracy = eval_model(test_dl, model,
                #                                             save_dir=None)
                results.append([str(x) + ',' + str(y), T, definition, auc_score, specificity, accuracy])
            # save numerical results

    results_patient_level_df = pd.DataFrame(results_patient_level,
                                            columns=['x,y', 'T', 'definition', 'auc', 'sepcificity', 'accuracy'])

    results_patient_level_df.to_csv(Output_results + train_test + '_patient_level_results.csv')
    result_df = pd.DataFrame(results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'accuracy'])
    result_df.to_csv(Output_results + train_test + '_results.csv')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    #data_folder = 'blood_only_data/'

    #eval_LSTM(constants.T_list, constants.xy_pairs, constants.FEATURES, data_folder, train_test='train', fake_test=False)

    data_folder_list = ['no_gcs/', 'all_cultures/', 'absolute_values/', 'strict_exclusion/']
    xy_pairs = [(24, 12)]
    for data_folder in data_folder_list:
        eval_LSTM([6], xy_pairs, constants.FEATURES, data_folder, train_test='train', fake_test=False)
