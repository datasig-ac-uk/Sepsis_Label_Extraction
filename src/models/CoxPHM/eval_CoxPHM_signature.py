import numpy as np


import sys
sys.path.insert(0, '../../../')

from definitions import *
from src.features.scaler import *
from src.features.dicts import *
from src.omni.functions import *
from src.models.CoxPHM.coxphm_functions import *


if __name__ == '__main__':
    x, y = 24, 12
    
    current_data_folder='full_culture_data/'    
    Root_Data,Model_Dir,Data_save=folders(current_data_folder) 
    
    Data_Dir =Root_Data+'experiments_'+str(x)+'_'+str(y)+'/test/'
    
    results = []
    for definition in definitions:
        for T in T_list:
            
            print('load dataframe and input features')
            df_sepsis_test = pd.read_pickle(Data_Dir + definition[1:] + '_dataframe.pkl')
            features_test = np.load(Data_Dir + 'james_features' + definition[1:] + '.npy')
            
            print('load test labels')
            labels_test = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

            # prepare dataframe for coxph model
            df_coxph_test = Coxph_df(df_sepsis_test, features_test, original_features, T, labels_test)

            # load trained coxph model
            cph = load_pickle(Model_Dir + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])

            # predict and evalute on test set
#             auc_score, specificity, accuracy = Coxph_eval(df_sepsis_test, cph, T,
#                                                           OUTPUT_DIR + 'predictions/' + 'coxphm_' +
#                                                           definition[1:] + '_' + str(T) + '.npy')
            auc_score, specificity, accuracy = Coxph_eval(df_coxph_test, cph, T,None)
            results.append([str(x) + ',' + str(y), T, definition, auc_score, specificity, accuracy])

    # save numerical results
    result_df = pd.DataFrame(results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'accuracy'])
    result_df.to_csv(Data_save+str(x) + '_' + str(y)+'_coxph_test_results.csv')