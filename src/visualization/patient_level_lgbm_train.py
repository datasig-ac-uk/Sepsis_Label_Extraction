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

        save=True
        current_data='blood_culture_data/'
        Root_Data,Model_Dir,Data_save=folders(current_data,model=MODELS[0])

        Data_Dir=Root_Data+'experiments_24_12/cv/'

        n=100
    
        model,k,a1='lgbm',5,6
        print("Now Collecting results from model", model)
        
        tprs_list,tnrs_list,fnrs_list,accs_list=[],[],[],[]


        results=[]
        
        for definition in definitions:
            print(definition)
            labels=np.load(Data_Dir+'label'+definition[1:]+'_6.npy')    
            probs_now=np.load(Data_Dir+'prob_preds'+definition[1:]+'_6.npy')

            
            _, _,_,_,val_full_indices=dataframe_cv_pack(None,k=k,\
                                                         definition=definition,\
                                                         path_save=Data_Dir,\
                                                         save=False)
        
            labels_true=labels_validation(labels, val_full_indices)
            

            CMs,_,_=suboptimal_choice_patient(labels_true, probs_now, val_full_indices, a1=a1,n=100)
            tprs, tnrs, fnrs,pres,accs = decompose_cms(CMs)
            
            results.append([definition,"{:.3f}".format(auc(1-tnrs,tprs)),\
                           "{:.3f}".format(output_at_metric_level(tnrs,tprs,metric_required=[0.85])),\
                           "{:.3f}".format(output_at_metric_level(accs,tprs,metric_required=[0.85]))])
            
        output_df= pd.DataFrame(results, columns=['definition','auc','sepcificity','accuracy'])
        
        if save:
            output_df.to_csv(Data_save+'lgbm_patient_level_train_results.csv')
        print(output_df)
