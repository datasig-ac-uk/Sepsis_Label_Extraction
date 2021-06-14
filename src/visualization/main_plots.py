import numpy as np
import pandas as pd
import random

import os
import pickle




import sys
sys.path.insert(0, '../')

import visualization.table_functions as table_functions
import visualization.plot_functions as plot_functions
import visualization.patientlevel_function as patientlevel
import features.mimic3_function as mimic3_myfunc
import constants as constants




if __name__ == '__main__':

    purpose = 'test'  # purpose can be 'test' or 'train'

    print("Collecting results on " + purpose+" set at setting x,y,T=24,12,6 from three models",
          constants.MODELS[0], constants.MODELS[1], constants.MODELS[-1])

    current_data = constants.exclusion_rules[0]
    Root_Data = constants.DATA_processed + current_data
    Root_Data, _, Output_predictions, Output_results = mimic3_myfunc.folders(
        current_data, model='LGBM')

    Data_Dir = Root_Data + purpose+'/'
    print("Labels will be collected from ", Data_Dir)
    print("The interim results will be collected from ",
          Output_predictions + purpose)
    
    Data_save_plots = constants.OUTPUT_DIR + 'plots/'
    mimic3_myfunc.create_folder(Data_save_plots)    

    Output_results=Output_results[:-5]

    Data_save_tables = constants.OUTPUT_DIR + 'tables/'
    mimic3_myfunc.create_folder(Data_save_tables)

    labels_list_list = []
    probs_list_list = []
    indices_list_list = []
    icuid_sequence_list_list = []

    x, y, a1 = 24, 12, 6
    n, n_bootstraps = 100, 100

    mean_fpr_list = [np.linspace(0, 1, 30)
                     for i in range(3)]  # for auc with CI plots

    for model in constants.MODELS:

        print("Now Collecting results from model", model)

        labels_list = []
        probs_list = []
        indices_list = []
        _, _, Output_predictions, _ = mimic3_myfunc.folders(
            current_data, model=model)

        for definition in constants.FEATURES:
            labels_now = np.load(Data_Dir + 'label_' + str(x) +
                                 '_' + str(y) + '_' + str(a1) + definition[1:] + '.npy')

            probs_now = np.load(
                Output_predictions + purpose+'/' + str(x) + '_' + str(y) + '_' + str(a1) + definition[1:] + '.npy') 

            icu_lengths_now = np.load(
                Data_Dir + 'icustay_lengths_' + str(x) + '_' + str(y) + definition[1:] + '.npy')
            icustay_fullindices_now = patientlevel.patient_idx(icu_lengths_now)

            labels_list.append(labels_now)
            probs_list.append(probs_now)
            indices_list.append(icustay_fullindices_now)

        labels_list_list.append(labels_list)
        probs_list_list.append(probs_list)
        indices_list_list.append(indices_list)
######################################  To produce FIgure 4, F.1, F.2 ###################################
    ######### Instance level auc plot/table #################
    print("--------------------------------Instance level now:--------------------------------")

    names = ['H1', 'H2', 'H3']
    mean_fpr_list = [np.linspace(0, 1, 30) for i in range(3)]

    print("----------- AUC plots from three models for real-time classification------------")
    print('Now 95% CI:')
    fprs_lists, tprs_lists = plot_functions.fprs_tprs_output(labels_list_list, probs_list_list,
                                                             n_bootstraps=n_bootstraps)

    error_list = plot_functions.CI_std_output(
        fprs_lists, tprs_lists, mean_fpr_list=mean_fpr_list)

    print("Plotting aucroc with CI from three models from real-time classification.")
    plot_functions.auc_subplots_errorbars(labels_list_list, probs_list_list, error_list, names=names,
                                          mean_fpr_list=mean_fpr_list, save_name=Data_save_plots+'auc_IC_plot_instance_level_three_models_'+purpose)

    print("Saving auc scores from three models for real-time classification.")
    table_functions.instance_level_auc_pd_threemodels(labels_list_list, probs_list_list,
                                                      pd_save_name=Data_save_tables + "auc_instance_level_three_models_"+purpose)

    ######### Patient level auc plots/tables #################
    print("--------------------------------Patient level now:--------------------------------")
    print('Now 95% CI:')
    fprs_lists_par, tprs_lists_par, labels_list_par, probs_list_par = plot_functions.fprs_tprs_output_patient_level(
        labels_list_list,
        probs_list_list,
        indices_list_list)
    print("Plotting patient-level aucroc with CI for three models.")
    error_list_par = plot_functions.CI_std_output(
        fprs_lists_par, tprs_lists_par, mean_fpr_list=mean_fpr_list)

    plot_functions.auc_subplots_errorbars(labels_list_par, probs_list_par, error_list_par, names=names,
                                          mean_fpr_list=mean_fpr_list, save_name=Data_save_plots+'auc_IC_plot_patient_level_three_models_'+purpose)

    print("A different way of getting tprs/fprs for three models.")
    tprs_list_list, fprs_list_list, fnrs_list_list, pres_list_list, \
        accs_list_list, time_list_list = patientlevel.patient_level_main_outputs_threemodels(labels_list_list,
                                                                                             probs_list_list,
                                                                                             indices_list_list,
                                                                                             n=n,
                                                                                             icuid_sequence_list_list=None)

    print("Patient level recall-precision plots for three models.")
    plot_functions.recall_specificity_subplots_patient_level(pres_list_list, tprs_list_list, names,
                                                             save_name=Data_save_plots + "recall_precision_plot_patient_level_three_models_"+purpose)

    print("Saving patient-level auc scores for three models.")
    table_functions.patient_level_auc_pd_threemodels(fprs_list_list, tprs_list_list, for_write=False,
                                                     pd_save_name=Data_save_tables + "auc_patient_level_three_models_"+purpose)

    print("Saving patient-level specificity at fixed sensitivity level 0.85 for three models.")
    table_functions.patient_level_output_pd_threemodels(fprs_list_list, tprs_list_list, metric_required=[0.85],
                                                        operator=lambda x: 1 - x, for_write=False,
                                                        pd_save_name=Data_save_tables + "specificity_patient_level_three_models_"+purpose)

    print("Saving patient-level accuracy at fixed sensitivity level 0.85 for three models.")
    table_functions.patient_level_output_pd_threemodels(accs_list_list, tprs_list_list, metric_required=[0.85],
                                                        operator=lambda x: x, for_write=False,
                                                        pd_save_name=Data_save_tables + "accuracy_patient_level_three_models_"+purpose)

######################################  To produce FIgure 7,8 ###################################
#     print("Now, for each fixed model, and for each of three definitions, producing instance/patient-level auc plots across four different xy pairs.")
#     for model in constants.MODELS:
#         plot_functions.auc_plot_xy_pairs(model=model, purpose='purpose')

######################################  To produce AUC FIgures of LSTM/COXPHM ###################################
    purpose = 'test'

    plot_functions.auc_plot_xy_pairs(Data_Dir, Data_save_plots, current_data=current_data)
######################################  To produce FIgure 5,6  ###################################
    print('produce sepsis onset time plots')
#     plot_functions.sepsis_onset_time_plots(
#         24, 12, 6, 'sensitivity', 0.85, 2000, save_dir=constants.OUTPUT_DIR + 'plots/')
