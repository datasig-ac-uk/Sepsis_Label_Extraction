import numpy as np
import torch
import random
from torch import nn, optim
import os
import features.mimic3_function as mimic3_myfunc
import omni.functions as omni_functions
import models.LSTM.LSTM_functions as lstm_functions
from models.nets import LSTM
from data.dataset import TimeSeriesDataset
import constants
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix,auc
import visualization.patientlevel_function as mimic3_myfunc_patientlevel


def train_LSTM(T_list, x_y, definitions, data_folder='blood_only/', fake_test=False):
    """

    :param T_list: (list of int) list of parameter T
    :param x_y: (list of int)list of sensitivity parameter x and y
    :param definitions:  (list of str) list of definitions. e.g.['t_suspision','t_sofa','t_sepsis_min']
    :param data_folder:   (str) folder name specifying
    :return:

    """
    results = []
    results_patient_level = []

    for x, y in x_y:
        data_folder = 'fake_test1/' + data_folder if fake_test else data_folder
        Root_Data, Model_Dir, Output_predictions, Output_results = mimic3_myfunc.folders(
            data_folder, model='LSTM')
        config_dir = constants.MODELS_DIR + 'blood_only/LSTM/hyperparameter/config'

        #     Data_Dir = Root_Data + '/processed/experiments_' + str(x) + '_' + str(y) + '/H3_subset/'
        Data_Dir = Root_Data + 'train' + '/'

        for definition in definitions:
            config = omni_functions.load_pickle(config_dir + definition[1:])
            print(config)

            #         for T in [6]:
            for T in T_list:
                print('load timeseries dataset')
                dataset = TimeSeriesDataset().load(Data_Dir + str(x) + '_' +
                                                   str(y) + definition[1:] + '_ffill.tsd')

                print('load train labels')
                labels_train = np.load(
                    Data_Dir + 'label'+'_'+str(x)+'_'+str(y)+'_'+str(T) + definition[1:] + '.npy')

                seed = 1023
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.backends.cudnn.deterministic = True

                # get torch dataloader for lstm
                train_dl, scaler = lstm_functions.prepared_data_train(
                    dataset, labels_train, True, 128, device)

                omni_functions.save_pickle(
                    scaler, Model_Dir + 'hyperparameter/scaler' + definition[1:])

                # specify lstm model architecture

                model = LSTM(in_channels=dataset.data.shape[-1], num_layers=1,
                             hidden_channels=config['hidden_channels'],
                             hidden_1=config['linear_channels'], out_channels=2,
                             dropout=0).to(device)

                lstm_functions.train_model(model, train_dl, n_epochs=config['epochs'],
                                           save_dir=Model_Dir + '_' +
                                           str(x) + '_' + str(y) + '_' +
                                           str(T) + definition[1:],
                                           loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=config['lr']))
                model.load_state_dict(
                    torch.load(Model_Dir + '_' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:],
                               map_location=torch.device('cpu')))
                auc_score, specificity, sensitivity, accuracy, true, preds = lstm_functions.eval_model(train_dl, model,
                                                                                          save_dir=Output_predictions+'train/' +
                                                                                                   str(x) + '_' + str(y) + '_'+ str(T) +
                                                                                                   definition[1:] + '.npy')

                df_sepsis = pd.read_pickle(
                    Data_Dir + str(x) + '_' + str(y) + definition[1:] + '_dataframe.pkl')
                preds = np.load(Output_predictions+'train/' + str(x) + '_' + str(y) + '_'+ str(T) + definition[1:] + '.npy')
                fpr, tpr, thresholds_ = roc_curve(true, preds, pos_label=1)

                index = np.where(tpr >= 0.85)[0][0]
                print(tpr[index])
                omni_functions.save_pickle(thresholds_[index], Model_Dir + 'thresholds/' +
                                           str(x) + '_' + str(y) + '_' +
                                           str(T) + definition[1:] + '_threshold.pkl')


                thresholds = np.arange(10000) / 10000
                CMs, _, _ = mimic3_myfunc_patientlevel.suboptimal_choice_patient_df(
                    df_sepsis, true, preds, a1=T, thresholds=thresholds, sample_ids=None)

                tprs, tnrs, fnrs, pres, accs = mimic3_myfunc_patientlevel.decompose_cms(CMs)
                threshold_patient = mimic3_myfunc_patientlevel.output_at_metric_level(thresholds, tprs,
                                                                                      metric_required=[0.85])

                omni_functions.save_pickle(threshold_patient, Model_Dir + 'thresholds_patients/' +
                                           str(x) + '_' + str(y) + '_' +
                                           str(T) + definition[1:] + '_threshold_patient.pkl')

                results_patient_level.append(
                    [str(x) + ',' + str(y), T, definition, "{:.3f}".format(auc(1 - tnrs, tprs)),
                     "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(
                         tnrs, thresholds, metric_required=[threshold_patient])), \
                     "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(
                         tprs, thresholds, metric_required=[threshold_patient])),
                     "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(accs, thresholds,
                                                                                       metric_required=[
                                                                                           threshold_patient]))])


                results.append([str(x) + ',' + str(y), T,
                                definition, auc_score, specificity,sensitivity, accuracy])

    result_df = pd.DataFrame(
            results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity','sensitivity', 'accuracy'])

    result_df.to_csv(Output_results +'train'+
                     '_results1.csv')
    ############Patient level now ###############

    results_patient_level_df = pd.DataFrame(results_patient_level,
                                                columns=['x,y', 'T', 'definition', 'auc', 'sepcificity', 'sensitivity',
                                                         'accuracy'])
    results_patient_level_df.to_csv(
        Output_results + 'train' + '_patient_level_results1.csv')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)


    T_list = constants.T_list

    data_folder = constants.exclusion_rules[0]
    x_y = constants.xy_pairs
    train_LSTM(T_list, x_y, constants.FEATURES,
                 data_folder)

    x_y = [(24, 12)]
    data_folder_list = constants.exclusion_rules[1:]
    for data_folder in data_folder_list:
        train_LSTM(T_list, x_y, constants.FEATURES,
                     data_folder)

