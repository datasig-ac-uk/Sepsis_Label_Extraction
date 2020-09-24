import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score

from src.features.scaler import *
from definitions import *
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

    results=[]
    for definition in ['t_sofa','t_suspicion','t_sepsis_min']:
        for T in [4,6,8,12]:
            print('load dataframe and input features')
            df_sepsis_test = pd.read_pickle(Save_Dir+definition[1:]+'_dataframe_test.pkl')
            features_test = np.load(Save_Dir + 'james_features_test' + definition[1:] + '.npy')


            rest_features = [str(i) for i in range(features_test.shape[1])[len(original_features):]]
            total_features = original_features + rest_features
            df2_test = pd.DataFrame(features_test,
                             columns=total_features)

            print('load test labels')
            scores = np.load(Save_Dir + 'scores' + definition[1:] + '_' + str(T) + '.npy')
            labels_test = np.load(Save_Dir + 'label_test' + definition[1:] + '_' + str(T) + '.npy')
            df_sepsis_test['SepsisLabel'] = labels_test

            df_coxph_test = Cox_model_input_output(df_sepsis_test,df2_test, T, True)

            #load trained coxph model
            cph = load_pickle(MODELS_DIR+'CoxPHM/'+str(x)+'_'+str(y)+'_'+str(T)+definition[1:])

            # predict and evalute on test set
            H_t = cph.predict_cumulative_hazard(df_coxph_test).iloc[T + 1, :]
            risk_score = 1 - np.exp(-H_t)
            df_coxph_test['risk_score'] = risk_score
            np.save(OUTPUT_DIR +'predictions/'+ 'coxphm_' + definition[1:] + '_' + str(T) + '.npy', df_coxph_test['risk_score'])
            fpr, tpr, thresholds = roc_curve(df_coxph_test['label'], df_coxph_test['risk_score'], pos_label=1)
            index = np.where(tpr >= 0.85)[0][0]
            auc_score, specificity = auc(fpr, tpr), 1 - fpr[index]

            test_pred_labels = (df_coxph_test['risk_score'] > thresholds[index]).astype('int')
            accuracy = accuracy_score(df_coxph_test['label'].values, test_pred_labels)
            print('auc, sepcificity,accuracy', auc_score, specificity, accuracy)
            results.append([str(x) + ',' + str(y), T, definition, auc_score, specificity, accuracy])
    print(results)
    result_df = pd.DataFrame(results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'accuracy'])
    result_df.to_csv(OUTPUT_DIR + 'numerical_results/'+'CoxPHM/' +str(x)+','+str(y)+"_results.csv")