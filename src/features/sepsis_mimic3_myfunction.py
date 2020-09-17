import numpy as np
import pandas as pd
from src.models.optimizers import *
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from src.features.dicts import *
import random
from sklearn.model_selection import KFold
import iisignature


def compute_icu(df_merge2, definition, return_results=False):
    nonsepsis_icu_number = len(sorted(list(df_merge2[(df_merge2[definition].isnull())].icustay_id.unique())))

    icu_number = len(sorted(list(df_merge2.icustay_id.unique())))
    print('ICUStay id number:', icu_number)

    print('Sepsis ICUStay number and Sepsis ICU ratio:', icu_number - nonsepsis_icu_number, \
          (icu_number - nonsepsis_icu_number) / icu_number)

    if return_results:
        return icu_number, icu_number - nonsepsis_icu_number, (icu_number - nonsepsis_icu_number) / icu_number


def partial_sofa(df):
    """ Partial reconstruction of the SOFA score from features available in the sepsis dataset. """
    # Init the tensor

    #   data = df.groupby('icustay_id', as_index=False).apply(lambda v: v.ffill())

    sofa = np.full([len(df)], np.nan)
    platelets = np.full([len(df)], np.nan)
    bilirubin_total = np.full([len(df)], np.nan)
    maps = np.full([len(df)], np.nan)
    creatinine = np.full([len(df)], np.nan)

    # Coagulation
    platelets_ = df['heart_rate']

    platelets[platelets_ >= 150] = 0
    platelets[(100 <= platelets_) & (platelets_ < 150)] = 1
    platelets[(50 <= platelets_) & (platelets_ < 100)] = 2
    platelets[(20 <= platelets_) & (platelets_ < 50)] = 3
    platelets[platelets < 20] = 4

    # Liver
    bilirubin_ = df['bilirubin_total']

    bilirubin_total[bilirubin_ < 1.2] = 0
    bilirubin_total[(1.2 <= bilirubin_) & (bilirubin_ <= 1.9)] = 1
    bilirubin_total[(1.9 < bilirubin_) & (bilirubin_ <= 5.9)] = 2
    bilirubin_total[(5.9 < bilirubin_) & (bilirubin_ <= 11.9)] = 3
    bilirubin_total[bilirubin_ > 11.9] = 4

    # Cardiovascular
    map_ = df['mean_airway_pressure']
    maps[map_ >= 70] = 0
    maps[map_ < 70] = 1

    # Creatinine
    creatinine_ = df['creatinine']
    creatinine[creatinine_ < 1.2] = 0
    creatinine[(1.2 <= creatinine_) & (creatinine_ <= 1.9)] = 1
    creatinine[(1.9 < creatinine_) & (creatinine_ <= 3.4)] = 2
    creatinine[(3.4 < creatinine_) & (creatinine_ <= 4.9)] = 3
    creatinine[creatinine_ > 4.9] = 4

    sofa = platelets + bilirubin_total + maps + creatinine

    return sofa


def dataframe_from_definition_discard(path_df, a1=6, a2=3, definition='t_sepsis_min'):
    """
        For specific definition among the three, deleting patients who has been diagnosied
        beofre a1 hours in icu and trimming valid patients' data such that all the data after
        a2 hours after sepsis onset according to definition will be deleted.


    nonsepsis==True: we keep nonsepsis data; otherwsie, we do not keep them

    """

    pd.set_option('display.max_columns', None)

    ##Reading df:

    df_sepsis = pd.read_csv(path_df, \
                            low_memory=False,
                            parse_dates=['admittime', 'dischtime', 'deathtime', 'intime', 'outtime',
                                         'starttime_first_vent', 'endtime_first_vent', 'charttime',
                                         't_suspicion', 't_sofa', 't_sepsis_min', 't_sepsis_max'])

    df_sepsis['floored_charttime'] = df_sepsis['charttime'].dt.floor('h')

    df_sepsis = df_sepsis.sort_values(['icustay_id', 'floored_charttime'])
    print('Initial size:', df_sepsis.deathtime.size)

    ### deleting rows that patients are dead already
    df_sepsis = df_sepsis[(df_sepsis['deathtime'] > df_sepsis['charttime']) | df_sepsis.deathtime.isnull()]
    df_sepsis['gender'] = df_sepsis['gender'].replace(('M', 'F'), (0, 1))

    ### Newly added by Lingyi to fill in the gaps
    df_sepsis_intime = df_sepsis[identifier + static_vars].sort_values(['icustay_id', 'floored_charttime']).groupby(
        ['icustay_id'], as_index=False).first()
    df_sepsis_intime['floored_intime'] = df_sepsis_intime['intime'].dt.floor('h')
    df_sepsis_intime2 = df_sepsis_intime[df_sepsis_intime['floored_charttime'] > df_sepsis_intime['floored_intime']]
    df_sepsis_intime2.drop(['floored_charttime'], axis=1, inplace=True)
    df_sepsis_intime2.rename(columns={"floored_intime": "floored_charttime"}, inplace=True)
    df_sepsis = pd.concat([df_sepsis, df_sepsis_intime2], ignore_index=True, sort=False).sort_values(
        ['icustay_id', 'floored_charttime'])

    ### Discarding patients developing sepsis within 4 hour in ICU
    df_sepsis = df_sepsis[(df_sepsis[definition].between(df_sepsis.intime + pd.Timedelta(hours=4), \
                                                         df_sepsis.outtime, inclusive=True)) \
                          | (df_sepsis[definition].isnull())]

    print("Size of instances after discarding patients developing sepsis within 4 hour in ICU:",
          df_sepsis.deathtime.size)

    df_first = df_sepsis[static_vars + categorical_vars + identifier].groupby(by=identifier, as_index=False).first()
    df_mean = df_sepsis[numerical_vars + identifier].groupby(by=identifier, as_index=False).mean()
    df_max = df_sepsis[flags + identifier].groupby(by=identifier, as_index=False).max()
    df_merge = df_first.merge(df_mean, on=identifier).merge(df_max, on=identifier)
    df_merge.equals(df_merge.sort_values(identifier))
    df_merge.set_index('icustay_id', inplace=True)

    ### Resampling
    df_merge2 = df_merge.groupby(by='icustay_id').apply(lambda x: x.set_index('floored_charttime'). \
                                                        resample('H').first()).reset_index()

    print("Size after averaging hourly measurement and resampling:", df_merge2.deathtime.size)

    df_merge2[['subject_id', 'hadm_id', 'icustay_id'] + static_vars] = df_merge2[
        ['subject_id', 'hadm_id', 'icustay_id'] + static_vars].groupby('icustay_id', as_index=False).apply(
        lambda v: v.ffill())

    ### Deleting cencored data after a2
    df_merge2 = df_merge2[
        ((df_merge2.floored_charttime - df_merge2[definition]).dt.total_seconds() / 60 / 60 < a2 + 1.0) \
        | (df_merge2[definition].isnull())]

    print("Size of instances after getting censored data:", df_merge2.deathtime.size)

    ### Adding icu stay since intime
    df_merge2['rolling_los_icu'] = (df_merge2['outtime'] - df_merge2['intime']).dt.total_seconds() / 60 / 60
    df_merge2['hour'] = (df_merge2['floored_charttime'] - df_merge2['intime']).dt.total_seconds() / 60 / 60

    ### Discarding patients staying less than 4 hour or longer than 20 days
    df_merge2 = df_merge2[(df_merge2['rolling_los_icu'] <= 20 * 24) & (4.0 <= df_merge2['rolling_los_icu'])]

    print("Size of instances after discarding patients staying less than 4 hour or longer than 20 days:",
          df_merge2.deathtime.size)

    ### Triming the data to icu instances only
    df_merge2 = df_merge2[df_merge2['floored_charttime'] <= df_merge2['outtime']]
    df_merge2 = df_merge2[df_merge2['intime'] <= df_merge2['floored_charttime']]
    print("After triming the data to icu instances:", df_merge2.deathtime.size)

    icustay_id_to_included_bool = (df_merge2.groupby(['icustay_id']).size() >= 4)
    icustay_id_to_included = df_merge2.icustay_id.unique()[icustay_id_to_included_bool]
    df_merge2 = df_merge2[df_merge2.icustay_id.isin(icustay_id_to_included)]

    ### Mortality check
    df_merge2['mortality'] = (~df_merge2.deathtime.isnull()).astype(int)

    print("Final size:", df_merge2.deathtime.size)

    return df_merge2


def cv_bundle(full_indices, shuffled_indices, k=5):
    train_patient_indices, train_full_indices, test_patient_indices, test_full_indices = [], [], [], []

    kf = KFold(n_splits=k)
    for a, b in kf.split(shuffled_indices):
        train_patient_indices.append(shuffled_indices[a])
        test_patient_indices.append(shuffled_indices[b])

        train_full_indices.append([full_indices[train_patient_indices[-1][i]] for i in range(len(a))])
        test_full_indices.append([full_indices[test_patient_indices[-1][i]] for i in range(len(b))])

    return train_patient_indices, train_full_indices, test_patient_indices, test_full_indices


def dataframe_cv_pack(df_sepsis, k=5, definition='t_sepsis_min', path_save='./', save=True):
    """

        Outputing the data lengths for each patient and outputing cv splitting as in James' setting.


        nonsepsis==True: we store/call the random indice with name
            './shuffled_indices'+definition[1:]+'.npy';
            otherwsie, it will/was named as
            './shuffled_indices'+definition[1:]+'_sepsisonly.npy'

    """
    if not save:

        icustay_lengths = np.load(path_save + 'icustay_lengths' + definition[1:] + '.npy')
        shuffled_indices = np.load(path_save + 'shuffled_indices' + definition[1:] + '.npy')
        total_indices = len(shuffled_indices)

    else:
        icustay_id = sorted(list(df_sepsis.icustay_id.unique()))
        total_indices = len(icustay_id)
        #         print('ICUStay id number:',total_indices)
        #         nonsepsis_icu_number=len(sorted(list(df_sepsis[(df_sepsis[definition].isnull())].icustay_id.unique())))
        #         print('ICUStay id number for nonsepsis:',nonsepsis_icu_number)

        icustay_lengths = list(df_sepsis.groupby('icustay_id').size())
        np.save(path_save + 'icustay_lengths' + definition[1:] + '.npy', icustay_lengths)

        shuffled_indices = np.arange(total_indices)
        random.seed(42)
        random.shuffle(shuffled_indices)
        np.save(path_save + 'shuffled_indices' + definition[1:] + '.npy', shuffled_indices)

    # lengths_cumsum is defined for getting the full indices
    icustay_lengths_cumsum = np.insert(np.cumsum(np.array(icustay_lengths)), 0, 0)
    icustay_fullindices = [np.arange(icustay_lengths[i]) + icustay_lengths_cumsum[i] for i in range(total_indices)]

    train_patient_indices, train_full_indices, test_patient_indices, test_full_indices = cv_bundle(icustay_fullindices, \
                                                                                                   shuffled_indices, \
                                                                                                   k=5)

    return icustay_lengths, train_patient_indices, train_full_indices, test_patient_indices, test_full_indices


def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2,
                               u_fp=-0.05, u_tn=0, check_errors=True, return_all_scores=False):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    if return_all_scores:
        return u
    else:
        return np.sum(u)


def convert_labels(df):
    """Convert the binary labels to the corresponding utility score.
    The labels are given binary 0-1 values, but are scored according to a pre-defined utility score. Here we convert the
    binary labels onto their corresponding utility value as this value will be more useful for prediction than the
    binary labels.
    """

    def conversion_function(labels):
        labels = labels['SepsisLabel']

        # Get same length zeros and ones
        zeros = np.zeros(shape=(len(labels)))
        ones = np.ones(shape=(len(labels)))

        # Get scores for predicting zero or 1
        zeros_pred = compute_prediction_utility(labels.values, zeros, return_all_scores=True)
        ones_pred = compute_prediction_utility(labels.values, ones, return_all_scores=True)

        # Input scores of 0 and 1
        scores = np.concatenate([zeros_pred.reshape(-1, 1), ones_pred.reshape(-1, 1)], axis=1)
        scores = pd.DataFrame(index=labels.index, data=scores, columns=[0, 1])

        # Make an overall utilty score equal to one_score - zero_score which encodes the benefit of the 1 prediction
        scores['utility'] = scores[1] - scores[0]

        return scores

    scores = df.groupby('icustay_id').apply(conversion_function)

    return scores


def label_scores(df, a1=12, Data_Dir='./', definition='t_sepsis_min', save=True):
    print('Labeling:')
    a = list((df[definition] - df.floored_charttime).dt.total_seconds() / 60 / 60 <= a1)
    label = np.array([int(a[i]) for i in range(len(a))])
    if save:
        np.save(Data_Dir + 'label' + definition[1:] + '_' + str(a1) + '.npy', label)
    print('length of label', len(label))

    df['SepsisLabel'] = label

    scores = convert_labels(df)

    if save:
        np.save(Data_Dir + 'scores' + definition[1:] + '_' + str(a1) + '.npy', scores.values)
    print('Size of scores', scores.values.shape)

    return label, scores.values


def jamesfeature(df, Data_Dir='./', definition='t_sepsis_min', save=True):
    df['HospAdmTime'] = (df['admittime'] - df['intime']).dt.total_seconds() / 60 / 60

    print('Finally getting james feature:')
    james_feature = lgbm_jm_transform(df)

    if save:
        np.save(Data_Dir + 'james_features' + definition[1:] + '.npy', james_feature)

    print('Size of james feature for definition', james_feature.shape)

    return james_feature


def validation_treebased(model, dataset, labels, scores, \
                         train_patient_indices, train_full_indices,
                         test_patient_indices, test_full_indices, \
                         binary=True):
    """

        For chosen model, and for dataset after all necessary transforms, we conduct validation and testing simutaneously.


    """

    if binary:
        labels_true = np.empty((0, 1), int)
        train_preds = np.empty((0, 1), int)
        train_idxs = np.empty((0, 1), int)
        prob_preds = np.empty((0, 1), float)


    else:
        test_utilities = []
        labels_true = np.empty((0, 1), float)
        train_binary_preds = np.empty((0, 1), float)
        train_binary_true = np.empty((0, 1), float)
        train_preds = np.empty((0, 1), float)
        scores_true = np.empty((0, 3), float)
    k = len(train_full_indices)

    val_idx_collection = []

    for i in range(k):

        #        print('Now training on the', i, 'splitting')

        tra_dataset = dataset[np.concatenate(train_full_indices[i]), :]
        val_dataset = dataset[np.concatenate(test_full_indices[i]), :]

        tra_binary_labels = labels[np.concatenate(train_full_indices[i])]
        val_binary_labels = labels[np.concatenate(test_full_indices[i])]

        if binary:

            model.fit(X=tra_dataset, y=tra_binary_labels)

            predicted_prob = model.predict_proba(val_dataset)[:, 1]
            prob_preds = np.append(prob_preds, predicted_prob)
            train_idxs = np.append(train_idxs, np.concatenate(test_full_indices[i]))
            labels_true = np.append(labels_true, val_binary_labels)


        #            train_preds=np.append(train_preds,(predicted_prob>=0.5).astype('int'))

        #             test_utility=compute_utility(val_scores, predicted_prob, thresh)
        #             print('Utility score:',test_utility)
        #             test_utilities.append(test_utility)

        else:

            tra_utility_labels = scores[np.concatenate(train_full_indices[i]), -1]
            val_utility_labels = scores[np.concatenate(test_full_indices[i]), -1]

            tra_scores = scores[np.concatenate(train_full_indices[i]), :]
            val_scores = scores[np.concatenate(test_full_indices[i]), :]
            scores_true = np.append(scores_true, val_scores, axis=0)

            model.fit(X=tra_dataset, y=tra_utility_labels)
            val_pred = model.predict(val_dataset)
            train_preds = np.append(train_preds, val_pred)

            train_binary_true = np.append(train_binary_true, val_binary_labels)

            thresh = optimize_utility_threshold(model.predict(tra_dataset), scores=tra_scores)
            print("Threshold now:", thresh)
            test_utility = compute_utility(val_scores, val_pred, thresh)
            print('Utility score for : {:.3f}'.format(test_utility))

            test_utilities.append(test_utility)

            train_binary_preds = np.append(train_binary_preds, (val_pred > thresh) * 1)

    if binary:

        fpr, tpr, thresholds = roc_curve(labels_true, prob_preds, pos_label=1)
        index = np.where(tpr >= 0.85)[0][0]
        #   train_preds=np.array((prob_preds>=thresholds[index]).astype('int'))
        #    print(train_preds.shape)
        train_preds = np.append(train_preds, (prob_preds >= thresholds[index]).astype('int'))
        #        CM=confusion_matrix(labels_true,train_preds)
        #         print(CM)
        #         print(classification_report(labels_true,train_preds))
        #        print('auc and sepcificity',roc_auc_score(labels_true,prob_preds),fpr[len(fpr)-index])
        print('auc and sepcificity', roc_auc_score(labels_true, prob_preds), 1 - fpr[index])
        print('accuracy', accuracy_score(labels_true, train_preds))

        #         print(scores_true.shape)
        #         score_normal=scores_true[:,1][scores_true[:,1] > 0].sum()
        #         print('Utility score is:',np.sum(scores_true[np.arange(len(train_preds)),train_preds])/score_normal)

        return train_preds, prob_preds, labels_true, roc_auc_score(labels_true, prob_preds), 1 - fpr[
            index], accuracy_score(labels_true, train_preds)
    else:
        print('Average utility score is:', np.mean(np.array(test_utilities)))
        print(confusion_matrix(train_binary_true, train_binary_preds))
        print(classification_report(train_binary_true, train_binary_preds))

        return train_preds


def patient_level_performance_new(train_preds, labels_true, test_full_indices, k=5):
    true_septic_perpatient = np.empty((0, 1), int)
    preds_septic_perpatient = np.empty((0, 1), int)

    true_septic_time_perpatient = np.empty((0, 1), int)
    preds_septic_time_perpatient = np.empty((0, 1), int)

    tt = 0
    for i in range(k):
        #        print('Fold', i)

        current_length = np.concatenate(test_full_indices[i]).shape[0]

        current_patient_num = len(test_full_indices[i])

        current_length_perpatient = np.array([len(test_full_indices[i][j]) for j in range(current_patient_num)])

        current_length_cumsum = np.insert(np.cumsum(np.array(current_length_perpatient)), 0, 0)

        current_fullindices = [int(tt) + np.arange(current_length_perpatient[j]) + current_length_cumsum[j] for j in
                               range(current_patient_num)]

        tt += current_length

        current_preds_septic_perpatient = np.array(
            [int(len(np.where(train_preds[current_fullindices[j]] > 0)[0]) > 0) for j in range(current_patient_num)])
        preds_septic_perpatient = np.append(preds_septic_perpatient, current_preds_septic_perpatient)

        current_true_septic_perpatient = np.array(
            [int(len(np.where(labels_true[current_fullindices[j]] > 0)[0]) > 0) for j in range(current_patient_num)])

        true_septic_perpatient = np.append(true_septic_perpatient, current_true_septic_perpatient)

        current_preds_septic_time_perpatient = np.array(
            [np.where(train_preds[current_fullindices[j]] > 0)[0][0] for j in range(current_patient_num) \
             if current_true_septic_perpatient[j] != 0 and \
             len(np.where(train_preds[current_fullindices[j]] > 0)[0]) != 0])
        preds_septic_time_perpatient = np.append(preds_septic_time_perpatient, current_preds_septic_time_perpatient)

        current_true_septic_time_perpatient = np.array(
            [np.where(labels_true[current_fullindices[j]] > 0)[0][0] for j in range(current_patient_num) \
             if current_true_septic_perpatient[j] != 0 and \
             len(np.where(train_preds[current_fullindices[j]] > 0)[0]) != 0])
        true_septic_time_perpatient = np.append(true_septic_time_perpatient, current_true_septic_time_perpatient)

    #         index_sample=np.where(current_true_septic_perpatient==1)[0][-1]
    #         print('sample with true:',labels_true[current_fullindices[index_sample]])
    #         print('preds:',train_preds[current_fullindices[index_sample]])
    #         print('\n')
    #     print(confusion_matrix(true_septic_perpatient,preds_septic_perpatient))
    #     print('accuracy:',accuracy_score(true_septic_perpatient,preds_septic_perpatient))

    sepsis_time_difference = true_septic_time_perpatient - preds_septic_time_perpatient
    #     print('error: mean and std time to sepsis are', np.mean(np.abs(sepsis_time_difference)),\
    #           np.std(sepsis_time_difference))

    return accuracy_score(true_septic_perpatient, preds_septic_perpatient), \
           true_septic_time_perpatient, preds_septic_time_perpatient, \
           confusion_matrix(true_septic_perpatient, preds_septic_perpatient)


def get_list(x, m):
    if len(x) >= 6:
        return [[np.nan for j in range(m)] for i in range(m - 1)] + list(zip(*(x[i:] for i in range(m))))
    else:
        return [[np.nan for j in range(m)] for i in range(len(x))]


def LL(datapiece):
    a = np.repeat(datapiece.reshape(-1, 1), 2, axis=0)
    return np.concatenate((a[1:, ], a[:-1, :]), axis=1)


def signature(datapiece, s=iisignature.prepare(2, 3), \
              order=3, leadlag=True, logsig=True):
    if leadlag:
        if logsig:
            return np.asarray([iisignature.logsig(LL(datapiece[i, :]), s) for i in range(datapiece.shape[0])])
        else:

            return np.asarray([iisignature.sig(LL(datapiece[i, :]), order) for i in range(datapiece.shape[0])])
    else:

        return np.asarray([iisignature.sig(datapiece[i, :], order) for i in range(datapiece.shape[0])])


def lgbm_jm_transform(dataframe1, feature_dict=feature_dict, \
                      windows1=8, windows2=7, windows3=6, \
                      order=3, leadlag=True, logsig=True):
    """
        Directly replicating how James extracted features.
    """

    count_variables = feature_dict_james['laboratory'] + ['temp_celcius']

    #    print("Counts for ",count_variables)
    rolling_count = dataframe1.groupby('icustay_id').apply(lambda x: x[count_variables].rolling(windows1).count())
    newdata1 = np.asarray(rolling_count)

    ## current number 27

    #     vitals=dataframe1.groupby('icustay_id').apply(lambda x: x[feature_dict_james['vitals']].ffill())
    #     vitals=np.asarray(vitals)

    dataframe1 = dataframe1.groupby('icustay_id', as_index=False).apply(lambda v: v.ffill())

    dataframe1['bun_creatinine'] = dataframe1['bun'] / dataframe1['creatinine']
    dataframe1['partial_sofa'] = partial_sofa(dataframe1)
    dataframe1['shock_index'] = dataframe1['heart_rate'] / dataframe1['nbp_sys']

    newdata = []

    for variable in feature_dict_james['vitals']:
        #        print("2nd moment and max/min for ",variable)

        #         dataframe1[variable]=vitals[:,i].reshape(-1,1)

        temp_var = np.array(dataframe1.groupby('icustay_id').apply(lambda x: x[variable].rolling(windows2).var()))
        temp_mean = np.array(dataframe1.groupby('icustay_id').apply(lambda x: x[variable].rolling(windows2).mean()))

        newdata.append(temp_var + temp_mean ** 2)

        temp_max = np.array(dataframe1.groupby('icustay_id').apply(lambda x: x[variable].rolling(windows3).max()))
        temp_min = np.array(dataframe1.groupby('icustay_id').apply(lambda x: x[variable].rolling(windows3).min()))

        newdata.append(temp_max)
        newdata.append(temp_min)

    newdata = np.asarray(newdata).T
    ## current number 3*7=21
    ##total 21+27=48

    newdata = np.concatenate((newdata1, newdata), axis=1)
    #    print('before signature computing, the size of newly added feature:', newdata.shape)

    sig_features = ['heart_rate', 'nbp_sys', 'abp_mean'] + feature_dict_james['derived'][:-1]

    for j in range(len(sig_features)):

        variable = sig_features[j]

        #            print("rolling signature for ",variable)

        if leadlag:
            if logsig:
                siglength = iisignature.logsiglength(2, order)
            else:
                siglength = iisignature.siglength(2, order)
        else:
            siglength = iisignature.siglength(1, order)

        sigfeature = np.empty((0, siglength), float)

        ## To extract all rolling windows of each icustay_id
        rollingwindows = dataframe1.groupby('icustay_id').apply(lambda x: \
                                                                    get_list(x[variable], windows2))

        ## apply signature for all rolling windows of each icustay_id at once
        for jj in range(len(rollingwindows)):
            rolling_data = signature(np.asarray(rollingwindows.values[jj]), order=order, leadlag=leadlag, logsig=logsig)
            sigfeature = np.concatenate((sigfeature, rolling_data), axis=0)

        assert newdata.shape[0] == sigfeature.shape[0]

        newdata = np.concatenate((newdata, sigfeature), axis=1)

    ## current step number 5*5=25
    ##total so far 48+25=73

    # Currently, we finished adding new features, need to concat with original data
    ##concat original dataset

    #    print('Convert original features to numerics')
    original_features = dataframe1[feature_dict_james['vitals'] + feature_dict_james['laboratory'] \
                                   + feature_dict_james['derived'] + ['age', 'gender', 'hour', 'HospAdmTime']]
    original_features = np.asarray(original_features)

    ## current step number 7+26+3+4=40
    ##total so far 73+40=113

    finaldata = np.concatenate((original_features, newdata), axis=1)
    #  print(finaldata.shape)
    return finaldata


def lgbm_jm_model():
    lgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'colsample_bytree': 0.5884502388447466,
        'min_child_samples': 191,
        'min_child_weight': 32,
        'min_split_gain': 0.0,
        'subsample': 0.477870931,
        'subsample_for_bin': 200000,
        'num_leaves': 36,
        'reg_alpha': 100,
        'reg_lambda': 20,
        'n_jobs': -1,
        'random_state': 1,
    }

    clf = LGBMClassifier().set_params(**lgb_params).set_params(**{'min_child_samples': 199, \
                                                                  'min_child_weight': 38, \
                                                                  'num_leaves': 49})

    return clf


def labels_validation(labels, test_full_indices):
    """
        labels on validation sets

    """

    labels_true = np.empty((0, 1), int)
    k = len(test_full_indices)

    for i in range(k):
        val_binary_labels = labels[np.concatenate(test_full_indices[i])]
        labels_true = np.append(labels_true, val_binary_labels)

    return labels_true