import numpy as np
import pandas as pd
import random
import os
import dill, pickle

from lightgbm import LGBMClassifier


from src.features.sepsis_mimic3_myfunction import *
from src.features.LGBM.lgbm_functions import *


if __name__ == '__main__':
    
    Root='/data/processed/full_culture_data/'

    a2,k=0,5
    x,y=24,12
    
    Data_Dir=Root+'experiments_'+str(x)+'_'+str(y)+'/train/'
    print(Data_Dir)
   
    Data_save=Root+'results/'
    
    definitions=[ 't_sofa','t_suspicion', 't_sepsis_min']
    T_list=[12,8,6,4]
    
     
    df_path='/data/raw/training_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'

    results=[]
    
    for a1 in T_list:
        
        for definition in definitions:
        
            print(a1,definition)
        
            current_labels=np.load(Data_Dir+'label'+definition[1:]+'_'+str(a1)+'.npy')
            feature_data=np.load(Data_Dir+'james_features'+definition[1:]+'.npy')
            icustay_lengths=np.load(Data_Dir+'icustay_lengths'+definition[1:]+'.npy')
           
            tra_patient_indices,tra_full_indices,val_patient_indices,val_full_indices=
                    cv_pack(icustay_lengths,k=k,definition=definition,path_save=Data_Dir,save=False)
                
            with open(Data_save+'lgbm_best_paras'+definition[1:]+'.pkl', 'rb') as file:
                best_paras_=pickle.load(file)

            clf=LGBMClassifier(random_state=42).set_params(**best_paras_)
            _, prob_preds, _,auc,specificity,accuracy=model_validation(clf,feature_data,\
                                                                           current_labels,\
                                                                           tra_full_indices,\
                                                                           val_full_indices)

            
            np.save(Data_Dir+'prob_preds'+definition[1:]+'_'+str(a1)+'.npy',prob_preds)
            
            results.append([str(x)+','+str(y),a1,definition,auc,specificity,accuracy])
        
    result_df = pd.DataFrame(results, columns=['x,y','a1', 'definition', 'auc','speciticity','accuracy'])
    result_df.to_csv(Data_save+"lgbm_cv_results.csv")

 
