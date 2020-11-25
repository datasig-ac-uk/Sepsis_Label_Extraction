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
        Root_Data,Model_Dir,_,_,_=folders(current_data,model=MODELS[0])

        a1,a2,k=6,0,5
        x,y=24,12
        n_iter=500
        
        Data_Dir=Root_Data+'experiments_'+str(x)+'_'+str(y)+'/cv/'

        print(Data_Dir)

        model=LGBMClassifier(random_state=42)
 
        for definition in definitions:
        
            feature_loading_model_tuning(model, Data_Dir,Model_Dir,definition,\
                                         a1,grid_parameters,n_iter=n_iter,k=k,save=True)
                

