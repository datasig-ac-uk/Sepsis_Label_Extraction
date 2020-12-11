import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc

sys.path.insert(0, '../')

import constants

import features.sepsis_mimic3_myfunction as mimic3_myfunc
import visualization.sepsis_mimic3_myfunction_patientlevel_clean as patientlevel_clean

if __name__ == '__main__':

        save=True
        current_data='blood_culture_data/'
        Root_Data, Model_Dir, _, Output_predictions,Output_results = mimic3_myfunc.folders(current_data, model=constants.MODELS[0])

        Output_predictions_cv=Output_predictions+'cv/'
        Data_Dir=Root_Data+'experiments_24_12/cv/'

        n = 100
    
        model,k,x,y,a1='lgbm',5,24,12,6
        print("Now Collecting results from model", model)
        
        tprs_list,tnrs_list,fnrs_list,accs_list=[],[],[],[]


        results=[]
        
        for definition in constants.FEATURES:
            print(definition)
            labels=np.load(Data_Dir+'label'+definition[1:]+'_6.npy')    
            probs_now=np.load(Output_predictions_cv+'prob_preds_'+str(x)+'_'+str(y)+'_'+str(a1)+'_'+definition[1:]+'.npy')

            
            _, _,_,_,val_full_indices = \
                mimic3_myfunc.dataframe_cv_pack(None,k=k, definition=definition,
                                                path_save=Data_Dir, save=False)
        
            labels_true = mimic3_myfunc.labels_validation(labels, val_full_indices)
            

            CMs, _ , _ = patientlevel_clean.suboptimal_choice_patient(labels_true, probs_now, val_full_indices, a1=a1,n=100)
            tprs, tnrs, fnrs,pres,accs = patientlevel_clean.decompose_cms(CMs)
            
            results.append([definition,"{:.3f}".format(auc(1-tnrs,tprs)),\
                           "{:.3f}".format(patientlevel_clean.output_at_metric_level(tnrs,tprs,metric_required=[0.85])),\
                           "{:.3f}".format(patientlevel_clean.output_at_metric_level(accs,tprs,metric_required=[0.85]))])
            
        output_df= pd.DataFrame(results, columns=['definition','auc','sepcificity','accuracy'])
        
        if save:
            output_df.to_csv(Output_results+'lgbm_patient_level_cv_results.csv')
        print(output_df)
