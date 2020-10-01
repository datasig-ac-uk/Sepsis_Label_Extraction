import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score

from src.features.scaler import *
from definitions import *
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
    df['SepsisLabel'] = labels
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
    fpr, tpr, thresholds = roc_curve(df_coxph_test['label'], df_coxph_test['risk_score'], pos_label=1)
    index = np.where(tpr >= 0.85)[0][0]
    auc_score, specificity = auc(fpr, tpr), 1 - fpr[index]

    test_pred_labels = (df_coxph_test['risk_score'] > thresholds[index]).astype('int')
    accuracy = accuracy_score(df_coxph_test['label'].values, test_pred_labels)
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

if __name__ == '__main__':
    x, y = 24, 12
    Save_Dir = DATA_DIR + '/processed/experiments_' + str(x) + '_' + str(y) + '/'

    results = []
    for definition in ['t_sofa', 't_suspicion', 't_sepsis_min']:
        for T in [4, 6, 8, 12]:
            print('load dataframe and input features')
            df_sepsis_test = pd.read_pickle(Save_Dir + definition[1:] + '_dataframe_test.pkl')
            features_test = np.load(Save_Dir + 'james_features_test' + definition[1:] + '.npy')
            print('load test labels')
            labels_test = np.load(Save_Dir + 'label_test' + definition[1:] + '_' + str(T) + '.npy')

            # prepare dataframe for coxph model
            df_coxph_test = Coxph_df(df_sepsis_test, features_test, original_features, T, labels_test)

            # load trained coxph model
            cph = load_pickle(MODELS_DIR + 'CoxPHM/' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])

            # predict and evalute on test set
            auc_score, specificity, accuracy = Coxph_eval(df_sepsis_test, cph, T,
                                                          OUTPUT_DIR + 'predictions/' + 'coxphm_' +
                                                          definition[1:] + '_' + str(T) + '.npy')

            results.append([str(x) + ',' + str(y), T, definition, auc_score, specificity, accuracy])

    # save numerical results
    result_df = pd.DataFrame(results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'accuracy'])
    result_df.to_csv(OUTPUT_DIR + 'numerical_results/' + 'CoxPHM/' + str(x) + ',' + str(y) + "_results.csv")
