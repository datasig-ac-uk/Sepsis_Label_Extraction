from src.features.sepsis_mimic3_myfunction import *
from src.features.scaler import *
import os
from definitions import *
from lifelines import CoxPHFitter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, \
    roc_curve, auc
from sklearn import preprocessing
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


def Cox_model_input_output(df, feature_dict, a1, normalize):
    # construct output as reuiqred np structrued array format
    df['censor_hours'] = a1 + 1 - df.groupby('icustay_id')['SepsisLabel'].cumsum()

    # extract input features according to feature_dict
    df_coxph = df[np.concatenate([['icustay_id']] + [feature_dict_Coxph[item] for item in feature_dict_Coxph], axis=0)]
    # foward filling and then fill nan and inf with 0
    df_coxph1 = df_coxph.groupby(['icustay_id']).ffill().reset_index()
    df_coxph1.replace(np.inf, np.nan, inplace=True)
    df_coxph1.replace(-np.inf, np.nan, inplace=True)
    # print(np.where(df_coxph2==np.inf))
    df_coxph1 = df_coxph1.fillna(0)
    df_coxph1.drop('index', axis=1, inplace=True)
    if normalize:
        standard_scaler = preprocessing.MinMaxScaler()
        x_scaled = standard_scaler.fit_transform(df_coxph1.values)
        df_coxph1[:] = x_scaled
    df_coxph1['label'] = df['SepsisLabel'].values
    df_coxph1['censor_hours'] = df['censor_hours'].values
    return df_coxph1


# feature dictionary of coxph model
feature_dict_Coxph = {
    'vitals': ['heart_rate', 'o2sat', 'temp_celcius', 'nbp_sys', 'mean_airway_pressure', 'abp_dias', 'resp_rate'],
    'laboratory': ['baseexcess', 'bicarbonate', 'ast', 'fio2', 'ph', 'pco2', 'so2', 'bun', 'alkalinephos', 'calcium', \
                   'chloride', 'creatinine', 'bilirubin_direct', 'glucose', 'lactate', 'magnesium', 'phosphate', \
                   'potassium', 'bilirubin_total', 'tropinin_i', 'hematocrit', 'hemoglobin', 'wbc', 'ptt', \
                   'fibrinogen', 'platelets'],
    'demographics': ['age', 'gender']
}


path_df = DATA_DIR + '/raw/metavision_sepsis_data_10_09_20_sensitivity_24_12.csv'
#path_df = DATA_DIR + '/raw/metavision_sepsis_blood_only_data_11_09_20_sensitivity_24_12.csv'
Data_Dir = './Sep_experiments_24_12/'
model_name='coxph'
print('sensitity 24_12 definition')
results = []
for a1 in [4,6,8,12]:
    print('a1=',a1)
    for definition in ['t_suspicion', 't_sofa', 't_sepsis_min']:
        print(definition)
        df_sepsis1 = dataframe_from_definition_discard(path_df, definition=definition)

        try:
            scores = np.load(Data_Dir + 'scores' + definition[1:] + '_' + str(a1) + '.npy')
            labels = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(a1) + '.npy')
        except:
            labels, scores = label_scores(df_sepsis1, a1=a1, Data_Dir=Data_Dir, definition=definition, save=True)
        df_sepsis1['SepsisLabel'] = labels

        icustay_lengths, train_patient_indices, train_full_indices, test_patient_indices, test_full_indices = dataframe_cv_pack(
            df_sepsis1,
            k=5, definition=definition,
            path_save=Data_Dir, save=True)
        train_idxs = [np.concatenate(train_full_indices[i]) for i in range(len(train_full_indices))]
        test_idxs = [np.concatenate(test_full_indices[i]) for i in range(len(test_full_indices))]
        df_coxph = Cox_model_input_output(df_sepsis1, feature_dict_Coxph, a1, False)
        test_true, test_preds = [], []
        for i in range(5):
            x_train = df_coxph.iloc[train_idxs[i], :]
            x_test = df_coxph.iloc[test_idxs[i], :]
            cph = CoxPHFitter(penalizer=0.001).fit(x_train, duration_col='censor_hours', event_col='label',
                                               show_progress=True, step_size=0.3)
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
        test_preds_full = np.concatenate([test_preds[i].reshape(-1,1) for i in range(len(test_preds))])
        print(test_true_full.shape, test_preds_full.shape)
        fpr, tpr, thresholds = roc_curve(test_true_full , test_preds_full, pos_label=1)
        index=np.where(tpr>=0.85)[0][0]
        specificity = 1 - fpr[index]
        print('auc and sepcificity',auc(fpr, tpr),1-fpr[index])
        print('threshold for fixed 0.85 sensitity level is :',thresholds[index])
        np.save(Data_Dir + 'coxph_' + 'prob_preds' + definition[1:] + '_' + str(a1) + '.npy', test_preds_full)
        test_pred_labels= (test_preds_full > thresholds[index]).astype('int')
        precision, recall, f1_score, support = precision_recall_fscore_support(test_true_full, test_pred_labels,
                                                                       average='weighted')
        accuracy = accuracy_score(test_true_full, test_pred_labels)
        print(classification_report(test_true_full, test_pred_labels, digits=4))
        print(confusion_matrix(test_true_full, test_pred_labels))
        results.append(['24,12', a1, definition, auc(fpr, tpr), specificity, accuracy])

result_df = pd.DataFrame(results, columns=['x,y','a1', 'definition', 'auc','speciticity','accuracy'])
result_df.to_csv(Data_Dir+model_name+"_24_12_Sep_results.csv")

