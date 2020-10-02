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
    
    Data_Dir_train=Root+'experiments_'+str(x)+'_'+str(y)+'/train/'
    Data_Dir_test=Root+'experiments_'+str(x)+'_'+str(y)+'/test/'

    Data_save=Root+'results/'
    
    definitions=[ 't_sofa','t_suspicion', 't_sepsis_min']
    T_list=[12,8,6,4]
        
    results=[]
    
    for a1 in T_list:
            
        for definition in definitions:
        
            print(a1,definition)
            
            label_train=np.load(Data_Dir_train+'label'+definition[1:]+'_'+str(a1)+'.npy')
            feature_train=np.load(Data_Dir_train+'james_features'+definition[1:]+'.npy')

            label_test=np.load(Data_Dir_test+'label'+definition[1:]+'_'+str(a1)+'.npy')
            feature_test=np.load(Data_Dir_test+'james_features'+definition[1:]+'.npy')

                        
            with open(Data_save+'lgbm_best_paras'+definition[1:]+'.pkl', 'rb') as file:
                best_paras_=pickle.load(file)
                
            clf=LGBMClassifier(random_state=42).set_params(**best_paras_)
            
            _, prob_preds_test, auc,specificity,accuracy=model_training(clf,feature_train,feature_test,label_train,label_test)
        
            np.save(Data_Dir_test+probs+'lgbm_prob_preds'+definition[1:]+'_'+str(a1)+'.npy',prob_preds_test)
            results.append([str(x)+','+str(y),a1,definition,auc,specificity,accuracy])
        
    result_df = pd.DataFrame(results, columns=['x,y','a1', 'definition', 'auc','speciticity','accuracy'])
    result_df.to_csv(Data_Dir_test+"lgbm_test_results.csv")

        

 
