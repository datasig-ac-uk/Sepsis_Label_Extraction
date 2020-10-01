import numpy as np
from src.features.scaler import *
from definitions import *
from lifelines import CoxPHFitter
from src.features.dicts import *
from src.omni.functions import *


def Coxph_df(df, features, feature_dict, T, labels):
    """
    :param df:
    :param features:
    :param feature_dict:
    :param T:
    :param labels:
    :return:
    """
    df['SepsisLabel']=labels
    df['censor_hours'] = T + 1 - df.groupby('icustay_id')['SepsisLabel'].cumsum()

    # construct output as reuiqred np structrued array format
    rest_features = [str(i) for i in range(features.shape[1])[len(feature_dict):]]
    total_features = original_features + rest_features
    df2 = pd.DataFrame(features, columns=total_features)

    df2.replace(np.inf, np.nan, inplace=True)
    df2.replace(-np.inf, np.nan, inplace=True)
    df2 = df2.fillna(-1)

    df2['label'] = labels
    df2['censor_hours'] = df['censor_hours'].values

    return df2


original_features = feature_dict_james['vitals'] + feature_dict_james['laboratory'] \
                    + feature_dict_james['derived'] + ['age', 'gender', 'hour', 'HospAdmTime']

coxph_hyperparameter = {
    'regularize_sofa': 0.01,
    'step_size_sofa': 0.3,
    'regularize_suspicion': 0.01,
    'step_size_suspicion': 0.3,
    'regularize_sepsis_min': 0.01,
    'step_size_sepsis_min': 0.3,
}

if __name__ == '__main__':
    x, y = 24, 12
    Save_Dir = DATA_DIR + '/processed/experiments_' + str(x) + '_' + str(y) + '/'

    for definition in ['t_sofa', 't_suspicion', 't_sepsis_min']:
        for T in [4, 6, 8, 12]:
            print('load dataframe and input features')
            df_sepsis_train = pd.read_pickle(Save_Dir + definition[1:] + '_dataframe_train.pkl')
            features_train = np.load(Save_Dir + 'james_features' + definition[1:] + '.npy')

            print('load train labels')
            labels_train = np.load(Save_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

            # prepare dataframe for coxph model
            df_coxph_train = Coxph_df(df_sepsis_train, features_train, original_features, T, labels_train)

            # fit CoxPHM
            cph = CoxPHFitter(penalizer=coxph_hyperparameter['regularize' + definition[1:]]) \
                .fit(df_coxph_train, duration_col='censor_hours', event_col='label',
                     show_progress=True, step_size=coxph_hyperparameter['step_size' + definition[1:]])

            save_pickle(cph, MODELS_DIR + 'CoxPHM/' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])
