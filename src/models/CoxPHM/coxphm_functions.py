import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score

import sys
sys.path.insert(0, '../../../')

from definitions import *
from src.features.dicts import *



# def folders(current_data):
    
#     Root_Data=DATA_processed+current_data  
    
#     Model_Dir=MODELS_DIR+current_data+'CoxPHM/'    
#     create_folder(Model_Dir)
    
#     Data_save=Root_Data+'results/'
#     create_folder(Data_save)
    
#     return Root_Data,Model_Dir,Data_save


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

def Coxph_eval(df, model, T, save_dir=None):
    """
    :param df:
    :param model:
    :param T:
    :param save_dir:
    :return:
    """
    # predict and evalute on test set
    H_t = model.predict_cumulative_hazard(df).iloc[T + 1, :]
    risk_score = 1 - np.exp(-H_t)

    df['risk_score'] = risk_score
    if save_dir is not None:
        np.save(save_dir, df['risk_score'])
    fpr, tpr, thresholds = roc_curve(df['label'], df['risk_score'], pos_label=1)
    index = np.where(tpr >= 0.85)[0][0]
    auc_score, specificity = auc(fpr, tpr), 1 - fpr[index]

    test_pred_labels = (df['risk_score'] > thresholds[index]).astype('int')
    accuracy = accuracy_score(df['label'].values, test_pred_labels)
    print('auc, sepcificity,accuracy', auc_score, specificity, accuracy)

    return auc_score, specificity, accuracy

# feature dictionary for coxph model
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