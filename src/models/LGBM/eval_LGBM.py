import os
import sys

import numpy as np
import pandas as pd
from sklearn import metrics


sys.path.insert(0, '../../')
import constants
from data.dataset import TimeSeriesDataset
import omni.functions as omni_functions

import features.mimic3_function as mimic3_myfunc
import visualization.patientlevel_function as mimic3_myfunc_patientlevel
import models.LGBM.LGBM_functions as lgbm_functions

def eval_LGBM(T_list, x_y, definitions, data_folder, train_test='test', thresholds=np.arange(10000) / 10000, fake_test=False):
    """
    This function compute evaluation on trained CoxPHM model, will save the prediction probability and numerical results
    for both online predictions and patient level predictions in the outputs directoty.
    :param T_list :(list of int) list of parameter T
    :param x_y:(list of int)list of sensitivity parameter x and y
    :param definitions:(list of str) list of definitions. e.g.['t_suspision','t_sofa','t_sepsis_min']
    :param data_folder:(str) folder name specifying
    :param train_test:(bool)True: evaluate the model on train set, False: evaluate the model on test set
    :param thresholds:(np array) A discretized array of probability thresholds for patient level evaluation
    :return: save prediction probability of online predictions, and save the results
              (auc/specificity/accuracy) for both online predictions and patient level predictions
    """
    results = []
    results_patient_level= []
    
    data_folder = 'fake_test1/' + data_folder if fake_test else data_folder
#     config_dir = constants.MODELS_DIR + 'blood_only_data/LGBM/hyperparameter/config'
    Root_Data, Model_Dir, Output_predictions, Output_results = mimic3_myfunc.folders(data_folder)
    purpose=train_test
    Data_Dir = Root_Data + purpose + '/'                                                                              
    
    
    for x, y in x_y:
        
                                                                                                                                                                                        
        for a1 in T_list:
            for definition in definitions:
                
                print(x, y, a1, definition)
                
                label= np.load(Data_Dir + 'label_' +str(x)+'_'+str(y) +'_' + str(a1) + definition[1:] + '.npy')
                feature = np.load(Data_Dir + 'james_features_'+str(x)+'_'+str(y)+ definition[1:]+'.npy')
                
                model_dir=Model_Dir+str(x)+'_'+str(y)+'_'+str(a1)+definition[1:]+'.pkl'
                print('Trained model from dic:',model_dir)
                preds, prob_preds, auc, specificity, accuracy = lgbm_functions.model_training(model_dir, feature, label)
                
                mimic3_myfunc.create_folder(Output_predictions+purpose)                                                                        
                np.save(Output_predictions+purpose + '/prob_preds_' + str(x) + '_' + str(y) + '_' + str(a1) + definition[1:] + '.npy',prob_preds)
                
                results.append([str(x) + ',' + str(y), a1, definition, auc, specificity, accuracy])
              
                ############Patient level now ###############                                                                               
                                                                                                
                df_sepsis = pd.read_pickle(Data_Dir + str(x) + '_' + str(y)+definition[1:] + '_dataframe.pkl')                                                                              
                CMs, _, _ = mimic3_myfunc_patientlevel.suboptimal_choice_patient_df(df_sepsis, label, prob_preds, a1=a1, thresholds=thresholds,sample_ids=None)     
                                                                                                
                tprs, tnrs, fnrs, pres, accs = mimic3_myfunc_patientlevel.decompose_cms(CMs)

                results_patient_level.append(
                    [str(x) + ',' + str(y), a1, definition, "{:.3f}".format(metrics.auc(1 - tnrs, tprs)),
                     "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(tnrs, tprs, metric_required=[0.85])),
                     "{:.3f}".format(mimic3_myfunc_patientlevel.output_at_metric_level(accs, tprs, metric_required=[0.85]))])
                                                                                                
                ############################################  
    result_df = pd.DataFrame(results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'accuracy'])
    result_df.to_csv(Output_predictions + purpose +'/lgbm_'+purpose+'_results.csv')
    ############Patient level now ############### 

    results_patient_level_df = pd.DataFrame(results_patient_level,
                                            columns=['x,y', 'T', 'definition', 'auc', 'sepcificity', 'accuracy'])

    results_patient_level_df.to_csv(Output_results +  'lgbm_'+purpose+'_patient_level_results.csv')
    ############################################ 


if __name__ == '__main__':


    data_folder = 'blood_only/'

    eval_LGBM(constants.T_list, constants.xy_pairs, constants.FEATURES, data_folder,train_test='train',  fake_test=False)
#     data_folder_list = ['no_gcs/', 'all_cultures/', 'absolute_values/', 'strict_exclusion/']
#     xy_pairs = [(24, 12)]
#     for data_folder in data_folder_list:
#         eval_LSTM([6], xy_pairs, constants.FEATURES, data_folder, fake_test=False)
