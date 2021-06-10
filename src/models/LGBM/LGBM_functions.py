import pickle
import sys
import pandas as pd
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import joblib

sys.path.insert(0, '../../')
import features.mimic3_function as mimic3_myfunc
import visualization.patientlevel_function as mimic3_myfunc_patientlevel


################################### LGBM tuning/training ########################################
def feature_loading(Data_Dir, definition, a1, x=24, y=12, k=5, cv=True, save=True):
    current_labels = np.load(Data_Dir + 'label' + '_' +
                             str(x) + '_' + str(y) + '_' + str(a1) + definition[1:] + '.npy')
    feature_data = np.load(Data_Dir + 'james_features' +
                           '_' + str(x) + '_' + str(y) + definition[1:] + '.npy')
    icustay_lengths = np.load(
        Data_Dir + 'icustay_lengths' + '_' + str(x) + '_' + str(y) + definition[1:] + '.npy')
    icustay_ids = np.load(Data_Dir + 'icustay_id' + '_' +
                          str(x) + '_' + str(y) + definition[1:] + '.npy')
    if cv:
        tra_patient_indices, tra_full_indices, val_patient_indices, val_full_indices = \
            mimic3_myfunc.cv_pack(
                icustay_lengths, k=k, definition=definition, path_save=Data_Dir, save=save)

        return current_labels, feature_data, tra_patient_indices, tra_full_indices, val_patient_indices, val_full_indices

    else:
        return current_labels, feature_data, icustay_lengths, icustay_ids


def model_validation(model, dataset, labels, tra_full_indices, val_full_indices):
    """

        For chosen model, and for dataset after all necessary transforms, we conduct k-fold cross-validation simutaneously.

        Input:
            model
            dataset: numpy version
            labels: numpy array
            tra_full_indices/val_full_indices: from cv splitting

        Output:
            tra_preds:  predicted labels on concatenated tra sets
            prob_preds: predicted risk scores for the predicted labels
            labels_true: true labels for tra_preds
            auc score
            sepcificity at fixed sensitivity level
            accuracy at fixed sensitivity level

    """

    labels_true = np.empty((0, 1), int)
    tra_preds = np.empty((0, 1), int)
    tra_idxs = np.empty((0, 1), int)
    prob_preds = np.empty((0, 1), int)

    k = len(tra_full_indices)

    val_idx_collection = []

    for i in range(k):
        print('Now training on the', i, 'splitting')

        tra_dataset = dataset[np.concatenate(tra_full_indices[i]), :]
        val_dataset = dataset[np.concatenate(val_full_indices[i]), :]

        tra_binary_labels = labels[np.concatenate(tra_full_indices[i])]
        val_binary_labels = labels[np.concatenate(val_full_indices[i])]

        model.fit(X=tra_dataset, y=tra_binary_labels)

        predicted_prob = model.predict_proba(val_dataset)[:, 1]
        prob_preds = np.append(prob_preds, predicted_prob)
        tra_idxs = np.append(tra_idxs, np.concatenate(val_full_indices[i]))
        labels_true = np.append(labels_true, val_binary_labels)

    fpr, tpr, thresholds = roc_curve(labels_true, prob_preds, pos_label=1)
    index = np.where(tpr >= 0.85)[0][0]

    tra_preds = np.append(
        tra_preds, (prob_preds >= thresholds[index]).astype('int'))
    print('auc and sepcificity', roc_auc_score(
        labels_true, prob_preds), 1 - fpr[index])
    print('accuracy', accuracy_score(labels_true, tra_preds))

    return tra_preds, prob_preds, labels_true, \
           roc_auc_score(labels_true, prob_preds), \
           1 - fpr[index], accuracy_score(labels_true, tra_preds)

'''
grid_parameters = {  # LightGBM
    'n_estimators': [40, 70, 100, 200, 400, 500, 800],
    'learning_rate': [0.08, 0.1, 0.12, 0.05],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    'max_depth': [4, 5, 6, 7, 8],
    'num_leaves': [5, 10, 16, 20, 25, 36, 49],
    'reg_alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
    'reg_lambda': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
    'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4],
    'subsample': np.arange(10)[5:] / 12,
    'subsample_freq': [10, 20],
    'max_bin': [100, 250, 500, 1000],
    'min_child_samples': [49, 99, 159, 199, 259, 299],
    'min_child_weight': np.arange(30) + 20}
'''

grid_parameters = { # LightGBM
'n_estimators': [40, 70, 100, 200, 400, 500, 800],
'learning_rate': [0.08, 0.1, 0.12, 0.05],
'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
'max_depth': [4, 5, 6, 7, 8],
'num_leaves': [5, 10, 16, 20, 25, 36, 49],
'reg_alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
'reg_lambda': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4],
'max_bin': [100, 250, 500, 1000],
'min_child_samples': [49, 99, 159, 199, 259, 299],
'min_child_weight': np.arange(30) + 20}


def model_tuning(model, dataset, labels, tra_full_indices, val_full_indices, param_grid,
                 grid=False, n_iter=100, n_jobs=32, scoring='roc_auc', verbose=2):
    """

        For chosen base model, we conduct hyperparameter-tuning on given cv splitting.

        Input:
            model
            dataset: numpy version
            labels: numpy array
            tra_full_indices/val_full_indices: from cv splitting
            param_grid:for hyperparameter tuning

        Output:

            set of best parameters for base model


    """

    k = len(tra_full_indices)

    cv = [[np.concatenate(tra_full_indices[i]), np.concatenate(
        val_full_indices[i])] for i in range(k)]

    if grid:
        gs = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          n_jobs=n_jobs,
                          cv=cv,
                          scoring=scoring,
                          verbose=verbose)
    else:
        np.random.seed(42)
        gs = RandomizedSearchCV(model,
                                param_grid,
                                n_jobs=n_jobs,
                                n_iter=n_iter,
                                cv=cv,
                                scoring=scoring,
                                verbose=verbose,
                                random_state=42)

    fitted_model = gs.fit(X=dataset, y=labels)
    best_params_ = fitted_model.best_params_
    print(best_params_)

    print(roc_auc_score(
        labels, fitted_model.predict_proba(dataset)[:,1]))

    return best_params_


def feature_loading_model_tuning(model, Data_Dir, Model_Dir, definition, a1, grid_parameters, x=24, y=12, n_iter=1000,
                                 k=5,
                                 n_jobs=-1, scoring='roc_auc', save=True):
    current_labels, feature_data, _, tra_full_indices, _, val_full_indices = feature_loading(Data_Dir,
                                                                                             definition,
                                                                                             a1,
                                                                                             x=x,
                                                                                             y=y,
                                                                                             k=k,
                                                                                             save=save)

    lgbm_best_paras_ = model_tuning(model, feature_data, current_labels, tra_full_indices,
                                    val_full_indices, grid_parameters, n_iter=n_iter, n_jobs=n_jobs)

    with open(Model_Dir + 'lgbm_best_paras' + definition[1:] + '.pkl', 'wb') as file:
        pickle.dump(lgbm_best_paras_, file)


def feature_loading_model_validation(Data_Dir, Model_Dir, definition, a1, x=24, y=12, k=5, save=False):
    """

        features loading and model validating altogether for different culture

    """

    current_labels, feature_data, _, tra_full_indices, _, val_full_indices = feature_loading(Data_Dir,
                                                                                             definition,
                                                                                             a1,
                                                                                             x=x,
                                                                                             y=y,
                                                                                             k=5,
                                                                                             save=save)

    with open(Model_Dir + 'lgbm_best_paras' + definition[1:] + '.pkl', 'rb') as file:
        best_paras_ = pickle.load(file)

    clf = LGBMClassifier(random_state=42).set_params(**best_paras_)

    _, prob_preds, _, auc, specificity, accuracy = model_validation(clf, feature_data,
                                                                    current_labels,
                                                                    tra_full_indices,
                                                                    val_full_indices)

    return prob_preds, auc, specificity, accuracy


# def model_training(model, train_set,test_set, train_labels, test_labels):

#         """

#         For chosen model, conduct standard training and testing

#         Input:
#             model
#             train_set,test_set: numpy version
#             train_labels, test_labels: numpy array

#         Output:
#             test_preds:  predicted labels on test set (numpy array)
#             prob_preds_test: predicted risk scores for the predicted test labels (numpy array)

#             auc score
#             sepcificity at fixed sensitivity level
#             accuracy at fixed sensitivity level


#         """

#         model.fit(X=train_set,y=train_labels)
#         prob_preds_test=model.predict_proba(test_set)[:,1]


#         print('Test:')
#         fpr, tpr, thresholds = roc_curve(test_labels, prob_preds_test, pos_label=1)
#         index=np.where(tpr>=0.85)[0][0]
#         test_preds=np.array((prob_preds_test>=thresholds[index]).astype('int'))

#         print('auc and sepcificity',roc_auc_score(test_labels,prob_preds_test),1-fpr[index])
#         print('accuracy',accuracy_score(test_labels,test_preds))

#         return test_preds, prob_preds_test, roc_auc_score(test_labels,prob_preds_test),\
#                1-fpr[index],accuracy_score(test_labels,test_preds)

def model_training(model_dir, test_set, test_labels):
    """

    For chosen model, conduct standard training and testing

    Input:
        model
        train_set,test_set: numpy version
        train_labels, test_labels: numpy array

    Output:
        test_preds:  predicted labels on test set (numpy array)
        prob_preds_test: predicted risk scores for the predicted test labels (numpy array)

        auc score
        sepcificity at fixed sensitivity level
        accuracy at fixed sensitivity level



    """

    model = joblib.load(model_dir)
    prob_preds_test = model.predict_proba(test_set)[:, 1]

    print('Model fitting:')
    try:
        model_threshold_dir = model_dir[:-4] + '_threshold' + model_dir[-4:]
        threshold = joblib.load(model_threshold_dir)

        test_preds = np.array((prob_preds_test >= threshold).astype('int'))

        tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        print('auc,sepcificity,sensitivity', roc_auc_score(
            test_labels, prob_preds_test), specificity, sensitivity)
        print('accuracy', accuracy_score(test_labels, test_preds))

        return test_preds, prob_preds_test, roc_auc_score(test_labels, prob_preds_test), \
               specificity, sensitivity, accuracy_score(test_labels, test_preds)
    except:
        return prob_preds_test, roc_auc_score(test_labels, prob_preds_test)


def model_fit_saving(model, train_set, train_labels, save_name, Data_Dir, x=24, y=12, a1=6, definition='t_sofa',
                     thresholds=np.arange(10000) / 10000):
    """

    For chosen model, conduct standard training and testing

    Input:
        model
        train_set,test_set: numpy version
        train_labels, test_labels: numpy array

    """

    model.fit(X=train_set, y=train_labels)

    joblib.dump(model, save_name)

    prob_preds_train = model.predict_proba(train_set)[:, 1]

    print('Model fitting:')
    fpr, tpr, thresholds_ = roc_curve(train_labels, prob_preds_train, pos_label=1)

    index = np.where(tpr >= 0.85)[0][0]
    print(tpr[index])
    joblib.dump(thresholds_[index], save_name[:-4] + '_threshold' + save_name[-4:])

    df_sepsis = pd.read_pickle(
        Data_Dir + str(x) + '_' + str(y) + definition[1:] + '_dataframe.pkl')

    CMs, _, _ = mimic3_myfunc_patientlevel.suboptimal_choice_patient_df(
        df_sepsis, train_labels, prob_preds_train, a1=a1, thresholds=thresholds, sample_ids=None)

    tprs, tnrs, fnrs, pres, accs = mimic3_myfunc_patientlevel.decompose_cms(CMs)
    threshold_patient = mimic3_myfunc_patientlevel.output_at_metric_level(thresholds, tprs, metric_required=[0.85])

    joblib.dump(threshold_patient, save_name[:-4] + '_threshold_patient' + save_name[-4:])

