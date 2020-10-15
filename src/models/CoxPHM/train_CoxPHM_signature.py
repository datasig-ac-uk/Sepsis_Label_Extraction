import numpy as np
from lifelines import CoxPHFitter

import sys

sys.path.insert(0, '../../../')

from definitions import *
from src.features.scaler import *
from src.features.dicts import *
from src.omni.functions import *
from src.models.CoxPHM.coxphm_functions import *

if __name__ == '__main__':
    x, y = 24, 12

    current_data_folder = 'full_culture_data/'
    Root_Data, Model_Dir, _ = folders(current_data_folder)

    Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/train/'

    for definition in definitions:
        config = load_pickle(MODELS_DIR + '/CoxPHM/hyperparameter/config' + definition[1:])
        for T in T_list:
            print('load dataframe and input features')
            df_sepsis_train = pd.read_pickle(Data_Dir + definition[1:] + '_dataframe.pkl')
            features_train = np.load(Data_Dir + 'james_features' + definition[1:] + '.npy')

            print('load train labels')
            labels_train = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

            # prepare dataframe for coxph model
            df_coxph_train = Coxph_df(df_sepsis_train, features_train, original_features, T, labels_train)

            # fit CoxPHM
            cph = CoxPHFitter(penalizer=config['regularize']) \
                .fit(df_coxph_train, duration_col='censor_hours', event_col='label',
                     show_progress=True, step_size=config['step_size'])

            save_pickle(cph, Model_Dir + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])
