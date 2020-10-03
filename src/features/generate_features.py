import numpy as np
import pandas as pd
import iisignature
import os
import random


import sys
sys.path.insert(0, '../../')
from definitions import *
from src.features.sepsis_mimic3_myfunction import *

    
if __name__ == '__main__':

    x,y = 24,12    
    a2=0

    
    path_df_train='/scratch/mimiciii/training_data/further_split/train_'+str(x)+'_'+str(y)+'.csv'
    path_df_test='/scratch/mimiciii/training_data/further_split/val_'+str(x)+'_'+str(y)+'.csv'

#     path_df_train = DATA_DIR + '/raw/full_culture_data/train_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'   
#     path_df_test = DATA_DIR + '/raw/full_culture_data/test_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'
    

    Save_Dir_train =DATA_processed + 'full_culture_data/experiments_'+str(x)+'_'+str(y)+'/train/'    
    Save_Dir_test = DATA_processed + 'full_culture_data/experiments_'+str(x)+'_'+str(y)+'/test/'
    
    print('generate train/set features for sensitity ' +str(x)+'_'+str(y) + ' definition')
    
    featureset_generator(path_df_train,Save_Dir_train,x=x,y=y,a2=a2, definitions=definitions,T_list=T_list)
    featureset_generator(path_df_test,Save_Dir_test,x=x,y=y,a2=a2, definitions=definitions,T_list=T_list)

