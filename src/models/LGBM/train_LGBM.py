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
import models.LGBM.LGBM_functions as lgbm_func

if __name__ == '__main__':

    a2, k = 0, 5
    current_data = 'blood_only/'
    Root_Data, Model_Dir, _, _ = mimic3_myfunc.folders(current_data, model='LGBM')

    train_Dir = Root_Data + 'train/'
    for x, y in constants.xy_pairs:
        for definition in constants.FEATURES:
            for a1 in constants.T_list:
                print(x, y, a1, definition)
                labels, features, icustay_lengths, icustay_ids = lgbm_func.feature_loading(train_Dir, definition, \
                                                                                           a1, k=k, cv=False)

                with open(Model_Dir + 'lgbm_best_paras' + definition[1:] + '.pkl', 'rb') as file:
                    best_paras_ = pickle.load(file)

                clf = LGBMClassifier(random_state=42).set_params(**best_paras_)

                model_dir = Model_Dir + str(x) + '_' + str(y) + '_' + str(a1) + definition[1:] + '.pkl'

                lgbm_func.model_fit_saving(clf, features, labels, model_dir)
