import numpy as np
import pandas as pd
import iisignature
import os
import random
import torch


from datetime import datetime, timedelta

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

from definitions import *
from src.data.dataset import TimeSeriesDataset
from src.data.functions import torch_ffill
from src.features.dicts import *
from src.features.sepsis_mimic3_myfunction import *

def dataset_generator(icustay_lengths,features):
        
        """
        Generating dataset for lstm from features
        
        """

        index = np.cumsum(np.array([0] + icustay_lengths))
        features_list = [torch.tensor(features[index[i]:index[i + 1]]) for i in range(index.shape[0] - 1)]
        column_list = [item for item in range(features.shape[1])]
        dataset = TimeSeriesDataset(data=features_list, columns=column_list, lengths=icustay_lengths)
        dataset.data = torch_ffill(dataset.data)
        dataset.data[torch.isnan(dataset.data)] = 0
        dataset.data[torch.isinf(dataset.data)] = 0
        
        return dataset

if __name__ == '__main__':

    x,y = 24,12
    
    definitions=['t_sofa','t_suspicion','t_sepsis_min']
    T_list=[4, 6, 8, 12]
    
    path_df_train = DATA_DIR + '/raw/full_culture_data/train_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'   
    path_df_test = DATA_DIR + '/raw/full_culture_data/test_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'
        
    Save_Dir_train = DATA_DIR + '/processed/full_culture_data/experiments_'+str(x)+'_'+str(y)+'/train/'
    create_folder(Save_Dir_train)
    
    Save_Dir_test = DATA_DIR + '/processed/full_culture_data/experiments_'+str(x)+'_'+str(y)+'/test/'
    create_folder(Save_Dir_test)
    
    print('generate features for sensitity ' +str(x)+'_'+str(y) + ' definition')
    
    results_train=[]
    results_test=[]
    
    for definition in definitions:
        
        a1,a2=6,0
        
        print('definition = '+str(definition))
        
        print('generate features on train and test sets')       
        df_sepsis1_train = dataframe_from_definition_discard(path_df_train, definition=definition,a1=a1,a2=a2)
        df_sepsis1_test = dataframe_from_definition_discard(path_df_test, definition=definition,a1=a1,a2=a2)
        
        print('save septic ratio for train and test sets')      
        icu_number,sepsis_icu_number, septic_ratio=compute_icu(df_sepsis1_train,definition,return_results=True)
        results_train.append([str(x)+','+str(y),definition,icu_number,sepsis_icu_number, septic_ratio])
        icu_number,sepsis_icu_number, septic_ratio=compute_icu(df_sepsis1_test,definition,return_results=True)
        results_test.append([str(x)+','+str(y),definition,icu_number,sepsis_icu_number, septic_ratio])
        
        print('save ICU Ids for train and test sets')       
        icuid_sequence_train=df_sepsis1_train.icustay_id.unique()
        np.save(Save_Dir_train +'icustay_id'+definition[1:]+'.npy',icuid_sequence_train)
        icuid_sequence_test=df_sepsis1_test.icustay_id.unique()
        np.save(Save_Dir_test +'icustay_id'+definition[1:]+'.npy',icuid_sequence_test)
        
        print('save ICU lengths for train and test sets')     
        icustay_lengths_train=list(df_sepsis1_train.groupby('icustay_id').size())
        np.save(Save_Dir_train +'icustay_lengths'+definition[1:]+'.npy',icustay_lengths_train)
        icustay_lengths_test=list(df_sepsis1_test.groupby('icustay_id').size())
        np.save(Save_Dir_test +'icustay_lengths'+definition[1:]+'.npy',icustay_lengths_test)

        print('save processed dataframe for lstm model')
        df_sepsis1_train.to_pickle(Save_Dir_train+definition[1:]+'_dataframe.pkl')
        df_sepsis1_test.to_pickle(Save_Dir_test + definition[1:]+'_dataframe.pkl')

        
        print('generate and save input features')
        features_train = jamesfeature(df_sepsis1_train, Data_Dir=Save_Dir_train, definition=definition)
        features_test = jamesfeature(df_sepsis1_test, Data_Dir=Save_Dir_test, definition=definition)

        print('generate and save timeseries dataset for LSTM model input')    
        dataset_train=dataset_generator(icustay_lengths_train,features_train)
        dataset_test=dataset_generator(icustay_lengths_test,features_test)
        
        dataset_train.save(Save_Dir_train + definition[1:] + '_ffill.tsd')
        dataset_test.save(Save_Dir_test + definition[1:] + '_ffill.tsd')
            
        print('gengerate and save labels')
        for T in T_list:
            print('T= ' + str(T))
            labels_train = label_generator(df_sepsis1_train, a1=T, Data_Dir=Save_Dir_train, definition=definition, save=True)
            labels_test = label_generator(df_sepsis1_test, a1=T, Data_Dir=Save_Dir_test, definition=definition, save=True)
            
    print('save icu spetic ratio to csv')               
    result_df_train = pd.DataFrame(results_train, columns=['x,y', 'definition', 'total_icu_no','sepsis_no','septic_ratio'])
    result_df_train.to_csv(Save_Dir_train+'icu_number.csv')     
    result_df_test = pd.DataFrame(results_test, columns=['x,y', 'definition', 'total_icu_no','sepsis_no','septic_ratio'])
    result_df_test.to_csv(Save_Dir_test+'icu_number.csv')
