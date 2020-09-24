import numpy as np
from src.features.scaler import *
from definitions import *
from lifelines import CoxPHFitter
from src.features.dicts import *
from src.omni.functions import *
def Cox_model_input_output(df, df2, a1, normalize=False):
    # construct output as reuiqred np structrued array format

    df['censor_hours'] = a1 + 1 - df.groupby('icustay_id')['SepsisLabel'].cumsum()

    df2.replace(np.inf, np.nan, inplace=True)
    df2.replace(-np.inf, np.nan, inplace=True)
    df2 = df2.fillna(-1)

    if normalize:
        standard_scaler = StandardScaler()
        x_scaled = standard_scaler.fit_transform(df2[df2.columns[1:]].values)
        df2[df2.columns[1:]] = x_scaled
    df2['label'] = df['SepsisLabel'].values
    df2['censor_hours'] = df['censor_hours'].values

    return df2

original_features = feature_dict_james['vitals'] + feature_dict_james['laboratory'] \
                        + feature_dict_james['derived'] + ['age', 'gender', 'hour', 'HospAdmTime']

coxph_hyperparameter= {
    'regularize_sofa':0.01,
    'step_size_sofa':0.3,
    'regularize_suspicion':0.01,
    'step_size_suspicion':0.3,
    'regularize_sepsis_min':0.01,
    'step_size_sepsis_min':0.3,
}

if __name__ == '__main__':
    x,y=24,12
    Save_Dir = DATA_DIR + '/processed/experiments_' + str(x) + '_' + str(y) + '/'


    for definition in ['t_sofa','t_suspicion','t_sepsis_min']:
        for T in [4,6,8,12]:
            print('load dataframe and input features')
            df_sepsis_train = pd.read_pickle(Save_Dir+definition[1:]+'_dataframe_train.pkl')
            features_train = np.load(Save_Dir + 'james_features' + definition[1:] + '.npy')


            rest_features = [str(i) for i in range(features_train.shape[1])[len(original_features):]]
            total_features = original_features + rest_features
            df2_train = pd.DataFrame(features_train,
                             columns=total_features)

            print('load train labels')
            scores = np.load(Save_Dir + 'scores' + definition[1:] + '_' + str(T) + '.npy')
            labels_train = np.load(Save_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')
            df_sepsis_train['SepsisLabel'] = labels_train

            df_coxph_train = Cox_model_input_output(df_sepsis_train, df2_train, T, True)
            cph = CoxPHFitter(penalizer=coxph_hyperparameter['regularize'+definition[1:]])\
               .fit(df_coxph_train, duration_col='censor_hours', event_col='label',
                    show_progress=True, step_size=coxph_hyperparameter['step_size'+definition[1:]])
            save_pickle(cph,MODELS_DIR+'CoxPHM/'+str(x)+'_'+str(y)+'_'+str(T)+definition[1:])


