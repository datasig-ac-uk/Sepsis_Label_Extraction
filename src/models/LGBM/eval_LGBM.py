import os

import numpy as np
import pandas as pd
from sklearn import metrics

import joblib


import models.LGBM.LGBM_functions as lgbm_functions
import visualization.patientlevel_function as mimic3_myfunc_patientlevel
import features.mimic3_function as mimic3_myfunc
import omni.functions as omni_functions
from data.dataset import TimeSeriesDataset
import constants as constants


def eval_LGBM(T_list, x_y, definitions, data_folder, train_test='test', thresholds=np.arange(10000) / 10000,
              fake_test=False):
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

    Root_Data, Model_Dir, Output_predictions, Output_results = mimic3_myfunc.folders(
        data_folder)
    purpose = train_test
    Data_Dir = Root_Data + purpose + '/'

    print(Data_Dir)
    for x, y in x_y:

        for a1 in T_list:
            for definition in definitions:

                print(x, y, a1, definition)

                label = np.load(Data_Dir + 'label_' + str(x) + '_' +
                                str(y) + '_' + str(a1) + definition[1:] + '.npy')
                feature = np.load(Data_Dir + 'james_features_' +
                                  str(x) + '_' + str(y) + definition[1:] + '.npy')

                model_dir = Model_Dir + \
                            str(x) + '_' + str(y) + '_' + str(a1) + definition[1:] + '.pkl'
                print('Trained model from dic:', model_dir)
                if data_folder == constants.exclusion_rules[0]:

                    preds, prob_preds, auc, specificity, sensitivity, accuracy = lgbm_functions.model_training(
                        model_dir, feature, label)
                    results.append([str(x) + ',' + str(y), a1,
                                    definition, auc, specificity, sensitivity, accuracy])
                else:
                    prob_preds, auc = lgbm_functions.model_training(
                        model_dir, feature, label)
                    results.append([str(x) + ',' + str(y), a1, definition, auc])
                mimic3_myfunc.create_folder(Output_predictions + purpose)
                np.save(Output_predictions + purpose + '/' + str(x) +
                        '_' + str(y) + '_' + str(a1) + definition[1:] + '.npy', prob_preds)

                ############Patient level now ###############
                if data_folder == constants.exclusion_rules[0]:
                    df_sepsis = pd.read_pickle(
                        Data_Dir + str(x) + '_' + str(y) + definition[1:] + '_dataframe.pkl')

                    CMs, _, _ = mimic3_myfunc_patientlevel.suboptimal_choice_patient_df(
                        df_sepsis, label, prob_preds, a1=a1, thresholds=thresholds, sample_ids=None)

                    tprs, tnrs, fnrs, pres, accs = mimic3_myfunc_patientlevel.decompose_cms(
                        CMs)

                    threshold_patient_dir = model_dir[:-4] + '_threshold_patient' + model_dir[-4:]
                    threshold_patient = joblib.load(threshold_patient_dir)

                    results_patient_level.append(
                        [str(x) + ',' + str(y), a1, definition, "{:.3f}".format(metrics.auc(1 - tnrs, tprs)),
                         "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(
                             tnrs, thresholds, metric_required=[threshold_patient])), \
                         "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(
                             tprs, thresholds, metric_required=[threshold_patient])),
                         "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(accs, thresholds,
                                                                                           metric_required=[
                                                                                               threshold_patient]))])

                ############################################
    if data_folder == constants.exclusion_rules[0]:
        result_df = pd.DataFrame(
            results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'sensitivity', 'accuracy'])
    else:
        result_df = pd.DataFrame(
            results, columns=['x,y', 'T', 'definition', 'auc'])

    result_df.to_csv(Output_results + purpose + '_results.csv') 
    ############Patient level now ###############
    if data_folder == constants.exclusion_rules[0]:
        results_patient_level_df = pd.DataFrame(results_patient_level,
                                                columns=['x,y', 'T', 'definition', 'auc', 'sepcificity', 'sensitivity',
                                                         'accuracy'])
        results_patient_level_df.to_csv(Output_results + purpose + '_patient_level_results.csv') 
    ############################################


if __name__ == '__main__':

    data_folder = constants.exclusion_rules[0]

    eval_LGBM(constants.T_list, constants.xy_pairs, constants.FEATURES,
              data_folder, train_test='test', fake_test=False)


    data_folder_list = constants.exclusion_rules[-2:]
    xy_pairs = [(24, 12)]
    for data_folder in data_folder_list:
        eval_LGBM([6], xy_pairs, constants.FEATURES, data_folder,train_test='test', fake_test=False)
