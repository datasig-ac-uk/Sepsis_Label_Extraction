import numpy as np
import pandas as pd
import random

import os
import pickle

from src.features.sepsis_mimic3_myfunction_new import *
from src.visualization.sepsis_mimic3_myfunction_patientlevel_clean import *
from src.visualization.plot_functions_clean import *
from src.visualization.table_functions_clean import *

if __name__ == '__main__':

    models=['lgbm','lstm','coxph']
    definitions=['t_sofa','t_suspicion','t_sepsis_min']
    titles=['LGBM','LSTM','CoxPHM']

    print("Collecting results from three models", titles[0],titles[1],titles[-1])
    
    Root='/data/processed/full_culture_data/'
    Data_Dir=Root+'24_12/test/'
    print("The interim results will be collected from ", Data_Dir)
    
    Data_save_plots='/data/processed/plots/"
    create_folder(Data_save_plot)
    
    Data_save_tables='/data/processed/tables/"  
    create_folder(Data_save_tables)
    
    labels_list_list=[]
    probs_list_list=[]
    test_indices_list_list=[]
    icuid_sequence_list_list=[]

    for model in range(len(models)):
        
        print("Now Collecting results from model", model)
        
        labels_list=[]
        probs_list=[]
        test_indices_list=[]
        icuid_sequence_list=[]

        
        for definition in definitions:
        
            labels_now=np.load(Data_Dir+'label'+definition[1:]+'_6.npy')    
            probs_now=np.load(Data_Dir+model+'_prob_preds'+definition[1:]+'_6.npy')
        
            icu_lengths_now=np.load(Data_Dir+'icustay_lengths'+definition[1:]+'.npy')        
            icustay_fullindices_now=patient_idx(icu_lengths_now)
        
            icuid_sequence_now=np.load(Data_Dir+'icustay_id'+definition[1:]+'.npy') 
                
            labels_list.append(labels_now)
            probs_list.append(probs_now)
            test_indices_list.append(icustay_fullindices_now)
            icuid_sequence_list.append(icuid_sequence_now)


        labels_list_list.append(labels_list)
        probs_list_list.append(probs_list)
        test_indices_list_list.append(test_indices_list)
        icuid_sequence_list_list.append(icuid_sequence_list)
    
    ######### Instance level auc plot/table #################
    Print("Instance level now:")
    
    names=['H1','H2','H3']
    Print("Instance level AUC plots for three models.")
    auc_subplots(labels_list_list,probs_list_list,names,save_name=Data_save_plots+"auc_plot_instance_level_three_models_test")
    
    Print("Saving instance-level auc scores for three models.")
    instance_level_auc_pd_threemodels(labels_list_list,probs_list_list,\
                                      pd_save_name=Data_save_tables+"auc_instance_level_three_models_test")
    
    ######### Patient level auc plots/tables #################
    Print("Patient level now:")
    tprs_list_list,fprs_list_list, fnrs_list_list,pres_list_list,\
    time_list_list, icuid_list_list=patient_level_main_outputs_threemodels(labels_list_list,\
                                                                           probs_list_list,\
                                                                           test_indices_list_list,\
                                                                           icuid_sequence_list_list)
    Print("Patient level AUC plots for three models.")
    auc_subplots_patient_level(fprs_list_list,tprs_list_list,names,titles=titles,\
                               save_name=Data_save_plots+"auc_plot_patient_level_three_models_test")
    
    Print("Patient level recall-precision plots for three models.")
    recall_specificity_subplots_patient_level(pres_list_list,tprs_list_list,names,\
                               save_name=Data_save_plots+"recall_precision_plot_patient_level_three_models_test")
    
    Print("Saving patient-level auc scores for three models.")
    patient_level_auc_pd_threemodels(fprs_list_list,tprs_list_list,for_write=False,\
                                     pd_save_name=Data_save_tables+"auc_patient_level_three_models_test")
    
    Print("Saving patient-level specificity at fixed sensitivity level 0.85 for three models.")
    patient_level_output_pd_threemodels(fprs_list_list,tprs_list_list,metric_required=[0.85],\
                                        operator=lambda x: 1-x, for_write=False,\
                                        pd_save_name=Data_save_tables+"specificity_patient_level_three_models_test")
    
    ######### Patient level early prediction performance #################    
    
    Print("Box plot (test set) of how far in advance sepsis casesbeing predicted correctly at sensitivity 85%")
    time_prediction_in_advance_list_list_tpr=patient_level_threded_output_threemodels(time_list_list,\
                                                                                  tprs_list_list,\
                                                                                  metric_required=[0.85])
    ylabel='Hour'
    name_seqs=names
    boxplots_prediction_time_inadvance(time_prediction_in_advance_list_list_tpr,\
                                       name_seqs, ylabel, savetitle=Data_save_plots+'boxplot_three_models_tpr_test')
    
    