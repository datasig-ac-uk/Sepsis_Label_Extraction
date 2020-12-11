import sys

import numpy as np

sys.path.insert(0, '../')

import constants

import features.sepsis_mimic3_myfunction as mimic3_myfunc
import visualization.sepsis_mimic3_myfunction_patientlevel_clean as patientlevel_clean
import visualization.plot_functions_clean as plot_functions_clean

if __name__ == '__main__':
    
    
    labels_list=[]
    probs_list=[]
    tprs_list=[]
    fprs_list=[]

    precision,n, a1=100,100,6
    definition='t_sepsis_min'
    
    current_data='blood_culture_data/'

    Root_Data,_,_,Output_predictions,Output_results=mimic3_myfunc.folders(current_data, model=constants.MODELS[0])
    
    for x,y in constants.xy_pairs:
    
        print(definition,x,y,model)
        Data_Dir=Root_Data+'experiments_'+str(x)+'_'+str(y)+'/test/'
        
    
        labels_now=np.load(Data_Dir+'label'+definition[1:]+'_6.npy')
    
        probs_now=np.load(Output_predictions+'prob_preds_'+str(x)+'_'+str(y)+'_'+str(a1)+'_'+definition[1:]+'.npy')
    
        icu_lengths_now=np.load(Data_Dir+'icustay_lengths'+definition[1:]+'.npy')        
    
        icustay_fullindices_now = patientlevel_clean.patient_idx(icu_lengths_now)
    
        tpr, fpr= patientlevel_clean.patient_level_auc(labels_now,probs_now,icustay_fullindices_now,precision,n=n,a1=a1)
    
        labels_list.append(labels_now)
        probs_list.append(probs_now)
        tprs_list.append(tpr)
        fprs_list.append(fpr)
    
    names=['48,24','24,12','12,6','6,3']
    
    plot_functions_clean.auc_plot(labels_list,probs_list,names=names,\
                     save_name=Output_results+'auc_plot_instance_level_'+model+'_sepsis_min_test') 
    plot_functions_clean.auc_plot_patient_level(fprs_list,tprs_list,names=names,\
                     save_name=Output_results+'auc_plot_patient_level_'+model+'_sepsis_min_test')    
    
 