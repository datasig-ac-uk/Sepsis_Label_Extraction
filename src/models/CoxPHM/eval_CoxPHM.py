import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc

import constants
import models.CoxPHM.coxphm_functions as coxphm_functions
import omni.functions as omni_functions
import features.mimic3_function as mimic3_myfunc
from visualization.patientlevel_function import decompose_cms, output_at_metric_level
from visualization.plot_functions import suboptimal_choice_patient
import visualization.patientlevel_function as mimic3_myfunc_patientlevel


def eval_CoxPHM(T_list, x_y, definitions, data_folder, train_test, signature,
                thresholds=np.arange(10000) / 10000, fake_test=False):
    """
    This function compute evaluation on trained CoxPHM model, will save the prediction probability and numerical results
    for both online predictions and patient level predictions in the outputs directoty.

    :param T_list :(list of int) list of parameter T
    :param x_y:(list of int)list of sensitivity parameter x and y
    :param definitions:(list of str) list of definitions. e.g.['t_suspision','t_sofa','t_sepsis_min']
    :param data_folder:(str) folder name specifying
    :param train_test:(bool)True: evaluate the model on train set, False: evaluate the model on test set
    :param signature:(bool) True:use the signature features + original features, False: only use original features
    :param thresholds:(np array) A discretized array of probability thresholds for patient level evaluation
    :return: save prediction probability of online predictions, and save the results
              (auc/specificity/accuracy) for both online predictions and patient level predictions
    """
    results = []
    results_patient_level = []
    model = 'CoxPHM' if signature else 'CoxPHM_no_sig'
    data_folder = 'fake_test1/' + data_folder if fake_test else data_folder
    Root_Data, Model_Dir, Output_predictions, Output_results = mimic3_myfunc.folders(
        data_folder, model=model)
    for x, y in x_y:

        Data_Dir = Root_Data + train_test + '/'

        for T in T_list:
            for definition in definitions:
                print('load dataframe and input features')
                df_sepsis = pd.read_pickle(
                    Data_Dir + str(x) + '_' + str(y) + definition[1:] + '_dataframe.pkl')
                features = np.load(
                    Data_Dir + 'james_features'+'_'+str(x) + '_' + str(y) + definition[1:] + '.npy')

                print('load test labels')
                labels = np.load(Data_Dir + 'label' + '_'+str(x) +
                                 '_'+str(y)+'_'+str(T) + definition[1:] + '.npy')

                # prepare dataframe for coxph model
                df_coxph = coxphm_functions.Coxph_df(df_sepsis, features,
                                                     coxphm_functions.original_features, T,
                                                     labels, signature=signature)

                # load trained coxph model
                cph = omni_functions.load_pickle(
                    Model_Dir + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])
                threshold = omni_functions.load_pickle(Model_Dir + 'thresholds/' +
                                                       str(x) + '_' + str(y) + '_' +
                                                       str(T) + definition[1:]+'_threshold.pkl')
                auc_score, specificity, sensitivity, accuracy = coxphm_functions.Coxph_eval1(df_coxph, cph, T, threshold,
                                                                                             Output_predictions + train_test + '/' +
                                                                                             str(x) + '_' + str(y) + '_' + str(T) +
                                                                                             definition[1:] + '.npy')
                preds = np.load(Output_predictions + train_test + '/' + str(x) + '_' + str(y) + '_' + str(T) +
                                definition[1:] + '.npy')
                CMs, _, _ = mimic3_myfunc_patientlevel.suboptimal_choice_patient_df(
                    df_sepsis, labels, preds, a1=T, thresholds=thresholds, sample_ids=None)

                tprs, tnrs, fnrs, pres, accs = mimic3_myfunc_patientlevel.decompose_cms(
                    CMs)

                threhold_patient = omni_functions.load_pickle(Model_Dir + 'thresholds_patients/' +
                                                              str(x) + '_' + str(y) + '_' +
                                                              str(T) + definition[1:] + '_threshold_patient.pkl')

                results_patient_level.append(
                    [str(x) + ',' + str(y), T, definition, "{:.3f}".format(auc(1 - tnrs, tprs)),
                     "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(
                         tnrs, thresholds, metric_required=[threhold_patient])),
                     "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(
                         tprs, thresholds, metric_required=[threhold_patient])),
                     "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(accs, thresholds,
                                                                                       metric_required=[
                                                                                           threhold_patient]))])

                results.append([str(x) + ',' + str(y), T,
                                definition, auc_score, specificity, sensitivity, accuracy])
            results_patient_level_df = pd.DataFrame(results_patient_level,
                                                    columns=['x,y', 'T', 'definition', 'auc', 'sepcificity',
                                                             'sensitivity', 'accuracy'])

            results_patient_level_df.to_csv(
                Output_results + train_test + '_patient_level_results.csv')
            result_df = pd.DataFrame(
                results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'sensitivity', 'accuracy'])
            result_df.to_csv(Output_results + train_test + '_results.csv')


if __name__ == '__main__':
    train_test = 'test'
    T_list = constants.T_list
    data_folder = constants.exclusion_rules[0]
    x_y = constants.xy_pairs
    eval_CoxPHM(T_list, x_y, constants.FEATURES, data_folder,
                train_test, True, fake_test=False)
    x_y = [(24, 12)]
    data_folder_list = constants.exclusion_rules[1:]
    for data_folder in data_folder_list:
        print(data_folder)
        eval_CoxPHM(T_list, x_y, constants.FEATURES, data_folder,
                    train_test, True, fake_test=False)
