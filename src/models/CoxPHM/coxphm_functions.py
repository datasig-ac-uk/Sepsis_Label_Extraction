import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.metrics import roc_curve, auc, accuracy_score

import sys
sys.path.insert(0, '../../../')

from definitions import *
from src.features.dicts import *
from src.features.sepsis_mimic3_myfunction import *
import ray
from ray import tune
from ray.tune.utils import pin_in_object_store, get_pinned_object

def folders(current_data):
    
    Root_Data=DATA_processed+current_data  
    
    Model_Dir=MODELS_DIR+current_data+'CoxPHM/'    
    create_folder(Model_Dir)
    
    Data_save=Root_Data+'results/'
    create_folder(Data_save)
    
    return Root_Data,Model_Dir,Data_save


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

search_space = {
    'regularize': tune.uniform(5e-2, 1e-3),
    'step_size': tune.uniform(0, 0.5)
}

def model_cv(config,data,a1):
    regularize, step_size = config['regularize'], config['step_size']
    df_coxph,train_full_indices,test_full_indices,k=get_pinned_object(data)
    test_true, test_preds = [], []
    train_idxs = [np.concatenate(train_full_indices[i]) for i in range(len(train_full_indices))]
    test_idxs = [np.concatenate(test_full_indices[i]) for i in range(len(test_full_indices))]
    try:
        for i in range(k):
             x_train = df_coxph.iloc[train_idxs[i], :]
             x_test = df_coxph.iloc[test_idxs[i], :]
             cph = CoxPHFitter(penalizer=regularize).fit(x_train, duration_col='censor_hours', event_col='label',
                                              show_progress=False, step_size=step_size)
             # cph.print_summary()
             H_t = cph.predict_cumulative_hazard(x_test).iloc[a1 + 1, :]
             risk_score = 1 - np.exp(-H_t)
             x_test['risk_score'] = risk_score
             fpr, tpr, thresholds = roc_curve(x_test['label'], x_test['risk_score'], pos_label=1)
             index = np.where(tpr >= 0.85)[0][0]
             print('auc and sepcificity', auc(fpr, tpr), 1 - fpr[index])
             test_true.append(x_test['label'].values)
             test_preds.append(x_test['risk_score'].values)
        test_true_full = np.concatenate([test_true[i].reshape(-1, 1) for i in range(len(test_true))])
        test_preds_full = np.concatenate([test_preds[i].reshape(-1, 1) for i in range(len(test_preds))])
        print(test_true_full.shape, test_preds_full.shape)
        fpr, tpr, thresholds = roc_curve(test_true_full, test_preds_full, pos_label=1)
        index = np.where(tpr >= 0.85)[0][0]
        specificity = 1 - fpr[index]
        print('auc and sepcificity', auc(fpr, tpr), 1 - fpr[index])
        auc_score=auc(fpr,tpr)
    except:
        auc_score=0
    tune.report(mean_accuracy=auc_score)