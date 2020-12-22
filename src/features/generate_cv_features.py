import numpy as np
import pandas as pd
import iisignature
import os
import random


import sys
sys.path.insert(0, '../')
import constants
import features.sepsis_mimic3_myfunction as mimic3_myfunc

    
if __name__ == '__main__':
    a2=0
    for x,y in constants.xy_pairs:     
    

        if x!=48:
                path_df_cv='/scratch/mimiciii/training_data/metavision_sepsis_blood_only_data_08_10_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'
        else:
            path_df_cv='/scratch/mimiciii/training_data/metavision_sepsis_blood_only_data_08_10_20.csv'
    

        Save_Dir_cv = constants.DATA_processed + 'blood_culture_data/experiments_'+str(x)+'_'+str(y)+'/cv/'    
    
        print('generate train/set features for sensitity ' +str(x)+'_'+str(y) + ' definition')
    
        mimic3_myfunc.featureset_generator(path_df_cv,Save_Dir_cv,x=x,y=y,a2=a2, definitions=constants.FEATURES,T_list=constants.T_list)

