import sys

sys.path.insert(0, '../../')

import models.LGBM.LGBM_functions as lgbm_func
import features.mimic3_function as mimic3_myfunc
import constants
import numpy as np
import pandas as pd
import random
import os
import pickle

from lightgbm import LGBMClassifier


def train_LGBM(T_list, x_y, definitions, data_folder, train_test='train', thresholds=np.arange(10000) / 10000,
              fake_test=False):
    results = []
    results_patient_level = []

    data_folder = 'fake_test1/' + data_folder if fake_test else data_folder

    Root_Data, Model_Dir, Output_predictions, Output_results = mimic3_myfunc.folders(
        data_folder)
    purpose = train_test
    Data_Dir = Root_Data + purpose + '/'

    print(Data_Dir)
    
    a2, k = 0, 5
    for x, y in x_y:
        for definition in definitions:
            for a1 in T_list:
                print(x, y, a1, definition)
                labels, features, icustay_lengths, icustay_ids = lgbm_func.feature_loading(Data_Dir, definition,
                                                                                           a1, k=k, x=x, y=y, cv=False)
                                                                                         
                with open(Model_Dir + 'lgbm_best_paras' + definition[1:] + '.pkl', 'rb') as file:
                    best_paras_ = pickle.load(file)

                clf = LGBMClassifier(random_state=42,n_jobs=4).set_params(**best_paras_)

                model_dir = Model_Dir + \
                            str(x) + '_' + str(y) + '_' + \
                            str(a1) + definition[1:] + '.pkl'
                        
                if data_folder == constants.exclusion_rules[0]:
                    
                    prob_preds_train,auc,spe, sen,acc, auc_patient,spe_patient,ses_patient,acc_patient=\
                lgbm_func.model_fit_saving(clf, features, labels, model_dir, Data_Dir, x=x, y=y, a1=a1,
                                           definition=definition,thresholds=thresholds)
                

                    results.append([str(x) + ',' + str(y), a1,
                                    definition, auc, spe, sen, acc])
                
                    results_patient_level.append(
                        [str(x) + ',' + str(y), a1, definition, "{:.3f}".format(auc_patient),
                         "{:.3f}".format(spe_patient),  "{:.3f}".format(ses_patient), "{:.3f}".format(acc_patient)])                    
                    
                else:
                    
                    prob_preds_train,auc=\
                lgbm_func.model_fit_saving(clf, features, labels, model_dir, Data_Dir, x=x, y=y, a1=a1,
                                           definition=definition,thresholds=[None])
                
                    results.append([str(x) + ',' + str(y), a1, definition, auc])
                    
                np.save(Output_predictions + 'train/' + str(x) +
                        '_' + str(y) + '_' + str(a1) + definition[1:] + '.npy', prob_preds_train)     

                ############################################
    if data_folder == constants.exclusion_rules[0]:
        result_df = pd.DataFrame(
            results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'sensitivity', 'accuracy'])
    else:
        result_df = pd.DataFrame(
            results, columns=['x,y', 'T', 'definition', 'auc'])

    result_df.to_csv(Output_predictions + purpose +
                     '/lgbm_' + purpose + '_results.csv')  ##to change?
    ############Patient level now ###############
    if data_folder == constants.exclusion_rules[0]:
        results_patient_level_df = pd.DataFrame(results_patient_level,
                                                columns=['x,y', 'T', 'definition', 'auc', 'sepcificity', 'sensitivity',
                                                         'accuracy'])
        results_patient_level_df.to_csv(Output_predictions + purpose +
                                        '/' + purpose + '_patient_level_results.csv')  ##to change?    

                
    
if __name__ == '__main__':
    
    
    data_folder = constants.exclusion_rules[0]

    train_LGBM(constants.T_list, constants.xy_pairs, constants.FEATURES,
              data_folder, train_test='train', fake_test=False)

    data_folders = constants.exclusion_rules[1:]
    for data_folder in data_folders:
        eval_LGBM(constants.T_list[2:3], constants.xy_pairs[1:2], constants.FEATURES,
              data_folder, train_test='train', fake_test=False)
