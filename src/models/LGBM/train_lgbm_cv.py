import numpy as np
import pandas as pd
import random
import os
import pickle

from lightgbm import LGBMClassifier

import sys
sys.path.insert(0, '../../../')
from definitions import *

from src.features.sepsis_mimic3_myfunction import *
from src.models.LGBM.lgbm_functions import *


if __name__ == '__main__':
    
    current_data='blood_culture_data/'
    Root_Data,Model_Dir,Data_save,Output_predictions,Output_results=folders(current_data,model=MODELS[0])
    
    Output_predictions_cv=Output_predictions+'cv/'
    create_folder(Output_predictions_cv)
    a2,k=0,5


    results=[]
    
    for x,y in xy_pairs:
        Data_Dir=Root_Data+'experiments_'+str(x)+'_'+str(y)+'/cv/'
        print(Data_Dir)

        for a1 in T_list:    
            for definition in definitions:
        
                print(x,y,a1,definition)       
        
                prob_preds,auc,specificity,accuracy=feature_loading_model_validation(Data_Dir,\
                                                                                     Model_Dir,\
                                                                                     definition,\
                                                                                     a1)
                
                 
                np.save(Output_predictions_cv+'prob_preds_'+str(x)+'_'+str(y)+'_'+str(a1)+'_'+definition[1:]+'.npy',prob_preds)
            
                results.append([str(x)+','+str(y),a1,definition,auc,specificity,accuracy])
        
    result_df = pd.DataFrame(results, columns=['x,y','a1','definition', 'auc','speciticity','accuracy'])
    result_df.to_csv(Output_results+"lgbm_cv_results.csv")

    main_result_tables(result_df,Data_save,model='lgbm',purpose='cv')
 
