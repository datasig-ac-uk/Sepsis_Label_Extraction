import sys

import numpy as np
from lifelines import CoxPHFitter
import pandas as pd

sys.path.insert(0, '../../')

import constants
import omni.functions as omni_functions
import models.CoxPHM.coxphm_functions as coxphm_functions

import features.sepsis_mimic3_myfunction as mimic3_myfunc


def train_CoxPHM(T_list, x_y, definitions, data_folder):
    """

    :param T_list(list of int): list of parameter T
    :param x_y(list of int):list of sensitivity parameter x and y
    :param definitions(list of str): list of definitions. e.g.['t_suspision','t_sofa','t_sepsis_min']
    :param data_folder(str): folder name specifying
    :return:
    """
    for x, y in x_y:
        Root_Data, Model_Dir, _ = mimic3_myfunc.folders(data_folder, model='CoxPHM_no_sig')

        Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/train/'
        for definition in definitions:
            config = omni_functions.load_pickle(constants.MODELS_DIR + 'hyperparameter/CoxPHM/config' + definition[1:])
            for T in T_list:
                print('load dataframe and input features')
                df_sepsis_train = pd.read_pickle(Data_Dir + definition[1:] + '_dataframe.pkl')
                features_train = np.load(Data_Dir + 'james_features' + definition[1:] + '.npy')

                print('load train labels')
                labels_train = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

                # prepare dataframe for coxph model
                df_coxph_train = coxphm_functions.Coxph_df(df_sepsis_train, features_train, coxphm_functions.original_features, T, labels_train,
                                                           signature=False)

                # fit CoxPHM
                cph = CoxPHFitter(penalizer=config['regularize']) \
                    .fit(df_coxph_train, duration_col='censor_hours', event_col='label',
                         show_progress=True, step_size=config['step_size'])

                omni_functions.save_pickle(cph, Model_Dir + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])


if __name__ == '__main__':
    x, y = 24, 12
    T_list = [4, 6, 8, 12]
    for x, y in [(6, 3), (12, 6), (24, 12), (48, 24)]:
        current_data_folder = 'blood_only_data/'
        Root_Data, Model_Dir, _ = mimic3_myfunc.folders(current_data_folder, model='CoxPHM_no_sig')

        Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/train/'

        for definition in constants.FEATURES:
            config = omni_functions.load_pickle(constants.MODELS_DIR + 'hyperparameter/CoxPHM/config' + definition[1:])
            for T in T_list:
                print('load dataframe and input features')
                df_sepsis_train = pd.read_pickle(Data_Dir + definition[1:] + '_dataframe.pkl')
                features_train = np.load(Data_Dir + 'james_features' + definition[1:] + '.npy')

                print('load train labels')
                labels_train = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

                # prepare dataframe for coxph model
                df_coxph_train = coxphm_functions.Coxph_df(df_sepsis_train, features_train, coxphm_functions.original_features, T, labels_train,
                                                           signature=False)

                # fit CoxPHM
                cph = CoxPHFitter(penalizer=config['regularize']) \
                    .fit(df_coxph_train, duration_col='censor_hours', event_col='label',
                         show_progress=True, step_size=config['step_size'])

                omni_functions.save_pickle(cph, Model_Dir + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])
