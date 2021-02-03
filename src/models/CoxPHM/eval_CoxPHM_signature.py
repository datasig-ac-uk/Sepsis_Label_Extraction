import sys
import numpy as np
import pandas as pd
from sklearn.metrics import auc

sys.path.insert(0, '../../')
import constants
import models.CoxPHM.coxphm_functions as coxphm_functions
import omni.functions as omni_functions
import features.sepsis_mimic3_myfunction as mimic3_myfunc
from src.visualization.sepsis_mimic3_myfunction_patientlevel_clean import decompose_cms, output_at_metric_level
from visualization.main_plots1 import suboptimal_choice_patient


def eval_CoxPHM(T_list, x_y, definitions, data_folder, train_test, signature,
                thresholds=np.arange(1000) / 1000, fake_test=False):
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
    Root_Data, Model_Dir, Data_save, Output_predictions, Output_results = mimic3_myfunc.folders(data_folder,
                                                                                                model=model)
    for x, y in x_y:

        Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/' + train_test + '/'

        for T in T_list:
            for definition in definitions:
                print('load dataframe and input features')
                df_sepsis = pd.read_pickle(Data_Dir + definition[1:] + '_dataframe.pkl')
                features = np.load(Data_Dir + 'james_features' + definition[1:] + '.npy')

                print('load test labels')
                labels = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

                # prepare dataframe for coxph model
                df_coxph = coxphm_functions.Coxph_df(df_sepsis, features,
                                                     coxphm_functions.original_features, T,
                                                     labels, signature=signature)

                # load trained coxph model
                cph = omni_functions.load_pickle(Model_Dir + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])

                # predict and evalute on test set
                auc_score, specificity, accuracy = coxphm_functions.Coxph_eval(df_coxph, cph, T,
                                                                               Output_predictions + str(x) + '_' + str(
                                                                                   y) + '_'
                                                                               + str(T) + definition[
                                                                                          1:] + '_' + train_test + '.npy')
                preds = np.load(Output_predictions + str(x) + '_' + str(y) + '_'
                                + str(T) + definition[1:] + '_' + train_test + '.npy')
                CMs, _, _ = suboptimal_choice_patient(df_sepsis, labels, preds, a1=6, thresholds=thresholds,
                                                      sample_ids=None)
                tprs, tnrs, fnrs, pres, accs = decompose_cms(CMs)
                print(output_at_metric_level(tnrs, tprs, metric_required=[0.85]))

                results_patient_level.append(
                    [str(x) + ',' + str(y), T, definition, "{:.3f}".format(auc(1 - tnrs, tprs)),
                     "{:.3f}".format(output_at_metric_level(tnrs, tprs, metric_required=[0.85])),
                     "{:.3f}".format(output_at_metric_level(accs, tprs, metric_required=[0.85]))])

                results.append([str(x) + ',' + str(y), T, definition, auc_score, specificity, accuracy])

        # save numerical results
    results_patient_level_df = pd.DataFrame(results_patient_level,
                                            columns=['x,y', 'T', 'definition', 'auc', 'sepcificity', 'accuracy'])

    results_patient_level_df.to_csv(Output_results + train_test + '_patient_level_results.csv')
    result_df = pd.DataFrame(results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'accuracy'])
    result_df.to_csv(Output_results + train_test + '_results.csv')


if __name__ == '__main__':
    train_test = 'train'

    #data_folder = 'blood_only_data/'
    #eval_CoxPHM(constants.T_list, constants.xy_pairs, constants.FEATURES, data_folder, train_test, signature=True, fake_test=False)

    data_folder_list = ['no_gcs/', 'all_cultures/', 'absolute_values/', 'strict_exclusion/']
    for data_folder in data_folder_list:
        eval_CoxPHM([6], [(24, 12)], constants.FEATURES, data_folder, train_test, signature=True, fake_test=False)
