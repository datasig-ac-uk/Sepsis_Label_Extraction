import numpy as np
import pandas as pd
import random

import os
import pickle

import sys
sys.path.insert(0, '../../')

from definitions import *

from src.features.sepsis_mimic3_myfunction import *
from src.visualization.sepsis_mimic3_myfunction_patientlevel_clean import *
from src.visualization.plot_functions_clean import *
from src.visualization.table_functions_clean import *

if __name__ == '__main__':
    
    
    labels_list=[]
    probs_list=[]
    tprs_list=[]
    fprs_list=[]

    precision,n, a1=100,100,6
    definition='t_sepsis_min'
    
    current_data='blood_culture_data/'
    model='lgbm'
    Root_Data,Model_Dir,Data_save=folders(current_data,model=MODELS[0])
    Data_save=Root_Data+'plots/'
    create_folder(Data_save)
    
    for x,y in xy_pairs:
    
        print(definition,x,y,model)
        Data_Dir=Root_Data+'experiments_'+str(x)+'_'+str(y)+'/test/'
        
    
        labels_now=np.load(Data_Dir+'label'+definition[1:]+'_6.npy')
    
        probs_now=np.load(Data_Dir+'lgbm_prob_preds'+definition[1:]+'_6.npy')
    
        icu_lengths_now=np.load(Data_Dir+'icustay_lengths'+definition[1:]+'.npy')        
    
        icustay_fullindices_now=patient_idx(icu_lengths_now)
    
        tpr, fpr= patient_level_auc(labels_now,probs_now,icustay_fullindices_now,precision,n=n,a1=a1)
    
        labels_list.append(labels_now)
        probs_list.append(probs_now)
        tprs_list.append(tpr)
        fprs_list.append(fpr)
    
    names=['48,24','24,12','12,6','6,3']
    
    auc_plot(labels_list,probs_list,names=names,\
                     save_name=Data_save+'auc_plot_instance_level_'+model+'_sepsis_min_test') 
    auc_plot_patient_level(fprs_list,tprs_list,names=names,\
                     save_name=Data_save+'auc_plot_patient_level_'+model+'_sepsis_min_test') 
    
    
 