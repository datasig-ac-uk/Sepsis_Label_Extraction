import numpy as np
import pandas as pd
import random
import os
import dill, pickle

from lightgbm import LGBMClassifier


from src.features.sepsis_mimic3_myfunction import *
from src.features.LGBM.lgbm_functions import *



if __name__ == '__main__':

        Root='/data/processed/LGBM/full_culture_data/'

        Data_save=Root+'results/'
        create_folder(Data_save)

        a1,a2,k=6,0,5
        x,y=24,12

        Data_Dir=Root+str(x)+'_'+str(y)+'/cv/'


        definitions=[ 't_sofa','t_suspicion', 't_sepsis_min']

        print(Data_Dir)

        model=LGBMClassifier(random_state=42)
        df_path='/data/raw/training_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'


        feature_generator(Data_Dir,df_path)


        for definition in definitions:
        
                current_labels=np.load(Data_Dir+'label'+definition[1:]+'_'+str(a1)+'.npy')
                feature_data=np.load(Data_Dir+'james_features'+definition[1:]+'.npy')
                icustay_lengths=np.load(Data_Dir+'icustay_lengths'+definition[1:]+'.npy')
           
                tra_patient_indices,tra_full_indices,val_patient_indices,val_full_indices=
                    cv_pack(icustay_lengths,k=k,definition=definition,path_save=Data_Dir,save=True)
        

                lgbm_best_paras_=model_tuning(model,feature_data, current_labels,tra_full_indices,\
                                      val_full_indices,grid_parameters, n_iter=1000)
        

                with open(Data_save+'lgbm_best_paras'+definition[1:]+'.pkl', 'wb') as file:
                        pickle.dump(lgbm_best_paras_, file)
                

