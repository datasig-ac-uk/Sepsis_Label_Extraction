import sys
import numpy as np
import pandas as pd


sys.path.insert(0, '../../')
import constants
import models.CoxPHM.coxphm_functions as coxphm_functions
import omni.functions as omni_functions
import features.mimic3_function as mimic3_myfunc
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix,auc
import visualization.patientlevel_function as mimic3_myfunc_patientlevel

def train_CoxPHM(T_list, x_y, definitions, data_folder, signature, fake_test):
    """

    :param T_list(list of int): list of parameter T
    :param x_y(list of int):list of sensitivity parameter x and y
    :param definitions(list of str): list of definitions. e.g.['t_suspision','t_sofa','t_sepsis_min']
    :param data_folder(str): folder name specifying
    :return:
    """
    results = []
    results_patient_level = []
    model = 'CoxPHM' if signature else 'CoxPHM_no_sig'
    config_dir = constants.MODELS_DIR + 'blood_only/CoxPHM/hyperparameter/config'
    data_folder = 'fake_test1/'+data_folder if fake_test else data_folder
    for x, y in x_y:

        Root_Data, Model_Dir, Output_predictions, Output_results = mimic3_myfunc.folders(
            data_folder, model=model)
        Data_Dir = Root_Data + 'train' + '/'
        for definition in definitions:

            config = omni_functions.load_pickle(config_dir + definition[1:])
            print(definition, config)
            for T in T_list:
                print(definition, x, y, T)
                print('load dataframe and input features')
                df_sepsis = pd.read_pickle(
                    Data_Dir + str(x) + '_' + str(y) + definition[1:] + '_dataframe.pkl')
                features_train = np.load(
                    Data_Dir + 'james_features'+'_'+str(x) + '_' + str(y) + definition[1:] + '.npy')

                print('load train labels')
                labels = np.load(
                    Data_Dir + 'label' + '_'+str(x)+'_'+str(y)+'_'+str(T) + definition[1:] + '.npy')

                # prepare dataframe for coxph model
                df_coxph = coxphm_functions.Coxph_df(df_sepsis, features_train,
                                                           coxphm_functions.original_features, T, labels,
                                                           signature=signature)

                # fit CoxPHM
                cph = coxphm_functions.CoxPHFitter(penalizer=config['regularize']) \
                    .fit(df_coxph, duration_col='censor_hours', event_col='label',
                         show_progress=True, step_size=config['step_size'])

                omni_functions.save_pickle(
                    cph, Model_Dir + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])
                auc_score, specificity, accuracy = coxphm_functions.Coxph_eval(df_coxph, cph, T,
                                                                               Output_predictions + 'train/' +
                                                                               str(x) + '_' + str(y) + '_' + str(T) +
                                                                               definition[1:] + '.npy')
                preds = np.load(Output_predictions + 'train/' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:] + '.npy')
                fpr, tpr, thresholds_ = roc_curve(labels, preds, pos_label=1)

                index = np.where(tpr >= 0.85)[0][0]
                print(tpr[index])
                omni_functions.save_pickle(thresholds_[index], Model_Dir +'thresholds/' +
                                           str(x) + '_' + str(y) + '_' +
                                           str(T) + definition[1:]+'_threshold.pkl')
                thresholds = np.arange(10000) / 10000
                CMs, _, _ = mimic3_myfunc_patientlevel.suboptimal_choice_patient_df(
                df_sepsis, labels, preds, a1=T, thresholds=thresholds, sample_ids=None)

                tprs, tnrs, fnrs, pres, accs = mimic3_myfunc_patientlevel.decompose_cms(CMs)
                thresholds = np.arange(10000) / 10000
                threshold_patient = mimic3_myfunc_patientlevel.output_at_metric_level(thresholds, tprs,
                                                                                      metric_required=[0.85])

                omni_functions.save_pickle(threshold_patient,Model_Dir + 'thresholds_patients/' +
                                           str(x) + '_' + str(y) + '_' +
                                           str(T) +definition[1:]+ '_threshold_patient.pkl')
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
                               definition, auc_score, specificity,0.85, accuracy])
    results_patient_level_df = pd.DataFrame(results_patient_level,
                                            columns=['x,y', 'T', 'definition', 'auc', 'sepcificity','sensitivity', 'accuracy'])

    results_patient_level_df.to_csv(
        Output_results + 'train' + '_patient_level_results1.csv')
    result_df = pd.DataFrame(
        results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity','sensitivity', 'accuracy'])
    result_df.to_csv(Output_results + 'train' + '_results1.csv')


if __name__ == '__main__':
    T_list = constants.T_list

    data_folder = constants.exclusion_rules[0]
    x_y = constants.xy_pairs
    train_CoxPHM(T_list, x_y, constants.FEATURES[1:],
                 data_folder, True, fake_test=False)

    x_y = [(24, 12)]
    data_folder_list = constants.exclusion_rules[1:]
    for data_folder in data_folder_list:
        train_CoxPHM(T_list, x_y, constants.FEATURES[1:],
                     data_folder, True, fake_test=False)
