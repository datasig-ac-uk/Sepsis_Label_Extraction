import numpy as np
import pandas as pd
import random

import os
import pickle

import os
import pickle
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '../')

import constants

import features.sepsis_mimic3_myfunction as mimic3_myfunc
import visualization.sepsis_mimic3_myfunction_patientlevel_clean as patientlevel_clean
import visualization.plot_functions_clean as plot_functions_clean
import visualization.table_functions_clean as table_functions_clean

if __name__ == '__main__':

    print("Collecting results at setting x,y,T=24,12,6 from three models",
          constants.MODELS[0], constants.MODELS[1], constants.MODELS[-1])

    current_data = 'blood_culture_data/'
    Root_Data = constants.DATA_processed + current_data

    Data_Dir = Root_Data + 'experiments_24_12/test/'
    print("The interim results will be collected from ", Data_Dir)

    Data_save_plots = Root_Data + 'plots/'
    mimic3_myfunc.create_folder(Data_save_plots)

    Data_save_tables = Root_Data + 'tables/'
    mimic3_myfunc.create_folder(Data_save_tables)

    labels_list_list = []
    probs_list_list = []
    test_indices_list_list = []
    icuid_sequence_list_list = []

    x, y, a1 = 24, 12, 6
    n, n_bootstraps = 100, 100

    mean_fpr_list = [np.linspace(0, 1, 30) for i in range(3)]  ### for auc with CI plots

    for model in constants.models:

        print("Now Collecting results from model", model)

        labels_list = []
        probs_list = []
        test_indices_list = []
        _, _, _, Output_predictions, _ = mimic3_myfunc.folders(current_data, model=model)

        for definition in constants.FEATURES:
            labels_now = np.load(Data_Dir + 'label' + definition[1:] + '_6.npy')
            probs_now = np.load(
                Output_predictions + 'prob_preds_' + str(x) + '_' + str(y) + '_' + str(a1) + '_' + definition[
                                                                                                   1:] + '.npy')

            icu_lengths_now = np.load(Data_Dir + 'icustay_lengths' + definition[1:] + '.npy')
            icustay_fullindices_now = patientlevel_clean.patient_idx(icu_lengths_now)

            labels_list.append(labels_now)
            probs_list.append(probs_now)
            test_indices_list.append(icustay_fullindices_now)

        labels_list_list.append(labels_list)
        probs_list_list.append(probs_list)
        test_indices_list_list.append(test_indices_list)

    ######### Instance level auc plot/table #################
    print("Instance level now:")

    names = ['H1', 'H2', 'H3']
    mean_fpr_list=[np.linspace(0, 1, 30) for i in range(3)]
    print("Instance level AUC plots for three models.")
    print('Now 95% CI:')
    fprs_lists, tprs_lists = plot_functions_clean.fprs_tprs_output(labels_list_list, probs_list_list,
                                                                   n_bootstraps=n_bootstraps)

    error_list = plot_functions_clean.CI_std_output(fprs_lists, tprs_lists, mean_fpr_list=mean_fpr_list)

    print("Plotting instance-level aucroc with CI for three models.")
    plot_functions_clean.auc_subplots_errorbars(labels_list_list,probs_list_list,error_list,names=names,\
                       mean_fpr_list=mean_fpr_list,save_name= Data_save_plots+'auc_plot_instance_level_three_models_test_errorbar_final')


    print("Saving instance-level auc scores for three models.")
    table_functions_clean.instance_level_auc_pd_threemodels(labels_list_list, probs_list_list, \
                                                            pd_save_name=Data_save_tables + "auc_instance_level_three_models_test")

    ######### Patient level auc plots/tables #################
    print("Patient level now:")

    fprs_lists_par, tprs_lists_par, labels_list_par, probs_list_par = plot_functions_clean.fprs_tprs_output_patient_level(
        labels_list_list, \
        probs_list_list, \
        test_indices_list_list)
    print("Plotting patient-level aucroc with CI for three models.")
    error_list_par = plot_functions_clean.CI_std_output(fprs_lists_par, tprs_lists_par, mean_fpr_list=mean_fpr_list)

    plot_functions_clean.auc_subplots_errorbars(labels_list_par,probs_list_par,error_list_par,names=names,\
                           mean_fpr_list=mean_fpr_list,save_name= Data_save_plots+'auc_plot_patient_level_three_models_test_errorbar_final')   


    print("A different way of getting tprs/fprs for three models.")
    tprs_list_list, fprs_list_list, fnrs_list_list, pres_list_list, \
    accs_list_list, time_list_list = patientlevel_clean.patient_level_main_outputs_threemodels(labels_list_list, \
                                                                                               probs_list_list, \
                                                                                               test_indices_list_list, \
                                                                                               n=n, \
                                                                                               icuid_sequence_list_list=None)

    print("Patient level recall-precision plots for three models.")
    plot_functions_clean.recall_specificity_subplots_patient_level(pres_list_list, tprs_list_list, names, \
                                                                   save_name=Data_save_plots + "recall_precision_plot_patient_level_three_models_test")

    print("Saving patient-level auc scores for three models.")
    table_functions_clean.patient_level_auc_pd_threemodels(fprs_list_list, tprs_list_list, for_write=False, \
                                                           pd_save_name=Data_save_tables + "auc_patient_level_three_models_test")

    print("Saving patient-level specificity at fixed sensitivity level 0.85 for three models.")
    table_functions_clean.patient_level_output_pd_threemodels(fprs_list_list, tprs_list_list, metric_required=[0.85], \
                                                              operator=lambda x: 1 - x, for_write=False, \
                                                              pd_save_name=Data_save_tables + "specificity_patient_level_three_models_test")

    print("Saving patient-level accuracy at fixed sensitivity level 0.85 for three models.")
    table_functions_clean.patient_level_output_pd_threemodels(accs_list_list, tprs_list_list, metric_required=[0.85], \
                                                              operator=lambda x: x, for_write=False, \
                                                              pd_save_name=Data_save_tables + "accuracy_patient_level_three_models_test")




