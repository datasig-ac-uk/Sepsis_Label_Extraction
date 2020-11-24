import numpy as np
from lifelines import CoxPHFitter

import sys

sys.path.insert(0, '../../../')

from definitions import *
from src.features.scaler import *
from src.features.dicts import *
from src.omni.functions import *
from src.models.CoxPHM.coxphm_functions import *

from src.features.sepsis_mimic3_myfunction import *


def train_CoxPHM(T_list, x_y, definitions, data_folder,signature):
    """
    Training on the CoxPHM model for specified T,x_y and definition parameters, save the trained model in model
    directory

    :param T_list(list of int): list of parameter T
    :param x_y(list of int):list of sensitivity parameter x and y
    :param definitions(list of str): list of definitions. e.g.['t_suspision','t_sofa','t_sepsis_min']
    :param data_folder(str): folder name specifying
    :param signature(bool): True:use the signature features + original features, False: only use original features
    :return: save trained model
    """
    for x, y in x_y:
        Root_Data, Model_Dir, _ = folders(data_folder, model='CoxPHM') if signature else \
            folders(data_folder, model='CoxPHM_no_sig')

        Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/train/'
        for definition in definitions:
            config = load_pickle(MODELS_DIR + 'hyperparameter/CoxPHM/config' + definition[1:])
            for T in T_list:
                print('load dataframe and input features')
                df_sepsis_train = pd.read_pickle(Data_Dir + definition[1:] + '_dataframe.pkl')
                features_train = np.load(Data_Dir + 'james_features' + definition[1:] + '.npy')

                print('load train labels')
                labels_train = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

                # prepare dataframe for coxph model
                df_coxph_train = Coxph_df(df_sepsis_train, features_train, original_features, T, labels_train,
                                          signature=signature)

                # fit CoxPHM
                cph = CoxPHFitter(penalizer=config['regularize']) \
                    .fit(df_coxph_train, duration_col='censor_hours', event_col='label',
                         show_progress=True, step_size=config['step_size'])

                save_pickle(cph, Model_Dir + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])


if __name__ == '__main__':

    x_y = [(6, 3), (12, 6), (24, 12), (48, 24)]
    T_list = [4, 6, 8, 12]
    data_folder = 'blood_only_data/'
    train_CoxPHM(T_list,x_y,definitions,data_folder,True)

