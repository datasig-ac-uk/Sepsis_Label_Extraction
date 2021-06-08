import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix,auc
import torch

from visualization.plot_functions import suboptimal_choice_patient

sys.path.insert(0, '../../')
from visualization.patientlevel_function import decompose_cms, output_at_metric_level
import features.mimic3_function as mimic3_myfunc
from models.nets import LSTM
import models.LSTM.LSTM_functions as lstm_functions
import omni.functions as omni_functions
from data.dataset import TimeSeriesDataset
import constants
import visualization.patientlevel_function as mimic3_myfunc_patientlevel
from multiprocessing import Pool


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
    config_dir = constants.MODELS_DIR + 'blood_only/LSTM/hyperparameter/config'
    Root_Data, Model_Dir, Output_predictions, Output_results = mimic3_myfunc.folders(data_folder,
                                                                                     model='LSTM')
    for x, y in x_y:

        Data_Dir = Root_Data + train_test + '/'

        #     Save_Dir = DATA_DIR + '/processed/experiments_' + str(x) + '_' + str(y) + '/H3_subset/'

        #         for T in [6]:
        for T in T_list:
            for definition in definitions:
                config = omni_functions.load_pickle(
                    config_dir + definition[1:])
                print('load timeseries dataset')
                dataset = TimeSeriesDataset().load(
                    Data_Dir + str(x) + '_' +
                    str(y) + definition[1:] + '_ffill.tsd')

                print('load train labels')
                labels = np.load(
                    Data_Dir + 'label' + '_' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:] + '.npy')
                # get torch dataloader for lstm
                scaler = omni_functions.load_pickle(
                    Model_Dir + 'hyperparameter/scaler' + definition[1:])
                #               scaler=load_pickle(MODELS_DIR+'/LSTM/hyperparameter/scaler'+definition[1:]+'H3_subset')
                test_dl = lstm_functions.prepared_data_test(
                    dataset, labels, True, scaler, 1000, device)

                # specify torch model architecture and load trained model
                model = LSTM(in_channels=dataset.data.shape[-1], num_layers=1,
                             hidden_channels=config['hidden_channels'],
                             hidden_1=config['linear_channels'], out_channels=2,
                             dropout=0).to(device)

                model.load_state_dict(
                    torch.load(Model_Dir + '_' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:],
                               map_location=torch.device('cpu')))



                auc_score, specificity, accuracy, true, preds = lstm_functions.eval_model(test_dl, model,
                                                                                          save_dir=Output_predictions+train_test+'/' + str(
                                                                                              x) + '_' + str(y) + '_'
                                                                                                   + str(
                                                                                              T) + definition[
                                                                                              1:] + '.npy')
                df_sepsis = pd.read_pickle(
                    Data_Dir + str(x) + '_' + str(y) + definition[1:] + '_dataframe.pkl')
                preds = np.load(Output_predictions+train_test+'/' + str(x) + '_' + str(y) + '_'+ str(T) + definition[1:] + '.npy')
                fpr, tpr, thresholds_ = roc_curve(true, preds, pos_label=1)

                index = np.where(tpr >= 0.85)[0][0]
                print(tpr[index])
                omni_functions.save_pickle(thresholds_[index], Model_Dir +'thresholds/' +
                                           str(x) + '_' + str(y) + '_' +
                                           str(T) + definition[1:]+'_threshold.pkl')
                CMs, _, _ = mimic3_myfunc_patientlevel.suboptimal_choice_patient_df(
                df_sepsis, true, preds, a1=T, thresholds=thresholds, sample_ids=None)

                tprs, tnrs, fnrs, pres, accs = mimic3_myfunc_patientlevel.decompose_cms(CMs)
                threshold_patient = mimic3_myfunc_patientlevel.output_at_metric_level(thresholds, tprs,
                                                                                      metric_required=[0.85])

                omni_functions.save_pickle(threshold_patient,Model_Dir + 'thresholds_patients/' +
                                           str(x) + '_' + str(y) + '_' +
                                           str(T) +definition[1:]+ '_threshold_patient.pkl')
                CMs, _, _ = suboptimal_choice_patient(df_sepsis, true, preds, a1=6, thresholds=thresholds,
                                                      sample_ids=None)
                tprs, tnrs, fnrs, pres, accs = decompose_cms(CMs)

                results_patient_level.append(
                    [str(x) + ',' + str(y), T, definition, "{:.3f}".format(auc(1 - tnrs, tprs)),
                     "{:.3f}".format(output_at_metric_level(
                         tnrs, tprs, metric_required=[0.85])),
                     "{:.3f}".format(output_at_metric_level(accs, tprs, metric_required=[0.85]))])

                # auc_score, specificity, accuracy = eval_model(test_dl, model,
                #                                             save_dir=None)
                results.append([str(x) + ',' + str(y), T,
                               definition, auc_score, specificity, accuracy])
            # save numerical results

    results_patient_level_df = pd.DataFrame(results_patient_level,
                                            columns=['x,y', 'T', 'definition', 'auc', 'sepcificity', 'accuracy'])

    results_patient_level_df.to_csv(
        Output_results + train_test + '_patient_level_results.csv')
    result_df = pd.DataFrame(
        results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'accuracy'])
    result_df.to_csv(Output_results + train_test + '_results.csv')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    seed = 1023
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    train_test = 'train'
    T_list = constants.T_list
    data_folder = constants.exclusion_rules1[0]
    x_y = constants.xy_pairs
    eval_LSTM(T_list, x_y, constants.FEATURES,
              data_folder, train_test, fake_test=False)

    x_y = [(24, 12)]
    data_folder_list = constants.exclusion_rules1[1:]
    for data_folder in data_folder_list:
        eval_LSTM(T_list, x_y, constants.FEATURES,
                  data_folder, train_test, fake_test=False)

