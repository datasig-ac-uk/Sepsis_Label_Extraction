import numpy as np
import pandas as pd
import random
import os
import pickle

from lightgbm import LGBMClassifier


import sys
sys.path.insert(0, '../../')
import constants

import features.sepsis_mimic3_myfunction as mimic3_myfunc
import models.LGBM.lgbm_functions as lgbm_func

if __name__ == '__main__':
    
    a2,k=0,5
    x,y,a1=24,12,6
    current_data='blood_culture_data/'
    Root_Data, Model_Dir, _, _, _ = mimic3_myfunc.folders(current_data, model='LGBM')

 
    train_Dir=Root_Data+'experiments_'+str(x)+'_'+str(y)+'/cv/'

    for definition in constants.FEATURES:
        
                print(x,y,a1,definition)
                labels,features,icustay_lengths, icustay_ids = lgbm_func.feature_loading(train_Dir,definition,\
                                                                                         a1,k=k, cv=False)

                                  
                with open(Model_Dir+'lgbm_best_paras'+definition[1:]+'.pkl', 'rb') as file:
                    best_paras_=pickle.load(file)
                
                clf=LGBMClassifier(random_state=42).set_params(**best_paras_)
                
                model_dir=Model_Dir+'lgbm_best_paras'+definition[1:]+'_trained_model_fake.pkl'
                
                lgbm_func.model_fit_saving(clf,features,labels, model_dir)


        

 
