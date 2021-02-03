import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

sys.path.insert(0, '../')
import features.dicts as dicts
import features.sepsis_mimic3_myfunction as mimic3_myfunc


################## Those for CV #############################

def probs_extraction(prob_preds, labels, val_full_indices, a1=6):
    """
        Input:

            1) instance level prob_preds on val_sets in one numpy array
            2) instance level labels on whole train set in one numpy array

                (the order of idxs in train set differs from that concantenated idx set of val sets)

        Output

          1) predicted probability sequence, namely, 'prob_preds_list'
          2) true labels per patient, namely, 'labels_list'

          on the validation sets in the form of

           [[XXX for patient 1 in validation set],[XXX for patient 2 in validation set],[],...]


          Also output
          3)sepsis label for patients, namely, 'true_septic_perpatient'

    """

    prob_preds_list = []
    labels_list = []

    true_septic_perpatient = np.empty((0, 1), int)
    length_perpatient = np.empty((0, 1), int)
    true_septic_lengths = np.empty((0, 1), int)

    tt = 0
    k = len(val_full_indices)

    for i in range(k):
        current_length = np.concatenate(val_full_indices[i]).shape[0]
        current_patient_num = len(val_full_indices[i])

        current_length_perpatient = np.array([len(val_full_indices[i][j]) for j in range(current_patient_num)])
        length_perpatient = np.append(length_perpatient, current_length_perpatient)

        current_length_cumsum = np.insert(np.cumsum(np.array(current_length_perpatient)), 0, 0)
        current_fullindices = [int(tt) + np.arange(current_length_perpatient[j]) + current_length_cumsum[j] for j in
                               range(current_patient_num)]

        tt += current_length

        prob_preds_list_current = [prob_preds[current_fullindices[j]] for j in range(current_patient_num)]
        prob_preds_list = prob_preds_list + prob_preds_list_current

        labels_list_current = [labels[val_full_indices[i][j]] for j in range(current_patient_num)]
        labels_list = labels_list + labels_list_current

        current_true_septic_perpatient = np.array(
            [int(len(np.where(labels[val_full_indices[i][j]] > 0)[0]) > 0) for j in range(current_patient_num)])
        true_septic_perpatient = np.append(true_septic_perpatient, current_true_septic_perpatient)

    return prob_preds_list, labels_list, true_septic_perpatient


def labels_validation(labels, val_full_indices):
    """
        Output labels on validation sets

    """

    labels_true = np.empty((0, 1), int)
    k = len(val_full_indices)

    for i in range(k):
        val_binary_labels = labels[np.concatenate(val_full_indices[i])]
        labels_true = np.append(labels_true, val_binary_labels)

    return labels_true


def patient_level_performance(val_preds, labels_true, val_full_indices, a1=6):
    """

        Mainly computing patient_level performance


        Inputs:
           val_preds: predicted labels on concatenated val sets
           labels_true: true corresponding labels
           val_full_indices:  [[val indices fold1],..,[val indices fold x],..]
                in each fold,
                    [val indices fold x]=[[indxs for patient 1 in fold x],..[indxs for patient j in fold x],....]
        Outputs:

          1) confusion matrix
          2) For the recognised septic patients, the time in advance we can predict sepsis
          3) predicted labels on val sets



    """

    true_septic_perpatient = np.empty((0, 1), int)
    preds_septic_perpatient = np.empty((0, 1), int)

    true_septic_time_perpatient = np.empty((0, 1), int)
    preds_septic_time_perpatient = np.empty((0, 1), int)

    k = len(val_full_indices)

    tt = 0

    for i in range(k):
        current_length = np.concatenate(val_full_indices[i]).shape[0]

        current_patient_num = len(val_full_indices[i])

        current_length_perpatient = np.array([len(val_full_indices[i][j]) for j in range(current_patient_num)])

        current_length_cumsum = np.insert(np.cumsum(np.array(current_length_perpatient)), 0, 0)

        current_fullindices = [int(tt) + np.arange(current_length_perpatient[j]) + current_length_cumsum[j] for j in
                               range(current_patient_num)]

        tt += current_length

        current_preds_septic_perpatient = np.array(
            [int(len(np.where(val_preds[current_fullindices[j]] > 0)[0]) > 0) for j in range(current_patient_num)])
        preds_septic_perpatient = np.append(preds_septic_perpatient, current_preds_septic_perpatient)

        current_true_septic_perpatient = np.array(
            [int(len(np.where(labels_true[current_fullindices[j]] > 0)[0]) > 0) for j in range(current_patient_num)])
        true_septic_perpatient = np.append(true_septic_perpatient, current_true_septic_perpatient)

        current_preds_septic_time_perpatient = np.array(
            [np.where(val_preds[current_fullindices[j]] > 0)[0][0] for j in range(current_patient_num) \
             if current_true_septic_perpatient[j] != 0 and \
             len(np.where(val_preds[current_fullindices[j]] > 0)[0]) != 0])
        preds_septic_time_perpatient = np.append(preds_septic_time_perpatient, current_preds_septic_time_perpatient)

        current_true_septic_time_perpatient = np.array(
            [np.where(labels_true[current_fullindices[j]] > 0)[0][0] for j in range(current_patient_num) \
             if current_true_septic_perpatient[j] != 0 and \
             len(np.where(val_preds[current_fullindices[j]] > 0)[0]) != 0])
        true_septic_time_perpatient = np.append(true_septic_time_perpatient, current_true_septic_time_perpatient)

    true_septic_time_perpatient += a1
    sepsis_time_difference = true_septic_time_perpatient - preds_septic_time_perpatient

    return confusion_matrix(true_septic_perpatient, preds_septic_perpatient), \
           sepsis_time_difference, preds_septic_perpatient

def patient_level_pred(df, true_labels, pred_labels, T, sample_ids=None, cohort='full'):
    """
    :param df :(dataframe) test dataframe
    :param true_labels :(array) array of true labels on test data
    :param pred_labels : (array) array of predicted labels on test data
    :param T: (int) left censor time
    :param sample_ids: (array) array of patient ids to take subset of patients, None if we use the whole set.
    :return: patient_true_label: (array) true labels at patient level
             patient_pred_label: (array) predicted labels at patient level
             CM : confusion matrix on the patient level prediction
             pred_septic_time :(array) predicted sepsis time for each patient
             true_septic_time: (array) true sepsis time for each patient
    """
    # construct data frame to store labels and predictions
    data = {'id': df['icustay_id'].values, 'labels': true_labels, 'preds': pred_labels}
    df_pred = pd.DataFrame.from_dict(data)
    if sample_ids is not None:
        df_pred = df_pred.loc[df_pred['id'].isin(sample_ids)]
    df_pred['rolling_hours'] = np.ones(df_pred.shape[0])
    df_pred['rolling_hours'] = df_pred.groupby('id')['rolling_hours'].cumsum()
    patient_icustay = df_pred.groupby('id')['rolling_hours'].max()
    patient_true_label = df_pred.groupby('id')['labels'].max()
    patient_pred_label = df_pred.groupby('id')['preds'].max()

    # get the predicted septic time and true septic time
    pred_septic_time = df_pred[df_pred['preds'] == 1].groupby('id')['rolling_hours'].min() - 1
    true_septic_time = df_pred[df_pred['labels'] == 1].groupby('id')['rolling_hours'].max() - 1
    ids = 0
    if cohort == 'correct_predicted':
        ids = df_pred[(df_pred['preds'] == 1) & (df_pred['labels'] == 1)]['id'].unique()
        df_pred1 = df_pred.loc[df_pred['id'].isin(ids)]
        pred_septic_time = df_pred1[df_pred1['preds'] == 1].groupby('id')['rolling_hours'].min() - 1
        true_septic_time = df_pred1[df_pred1['labels'] == 1].groupby('id')['rolling_hours'].min() - 1 + T

    return patient_true_label, patient_pred_label.values, \
           confusion_matrix(patient_true_label, patient_pred_label), \
           pred_septic_time, true_septic_time, ids, patient_icustay

def suboptimal_choice_patient_df(df, labels_true, prob_preds, a1=6, thresholds=np.arange(100)[1:-20] / 100,
                              sample_ids=None):
    """
        Finding suboptimal solution by through different threshold for probability
        Outputs:
          1) a list of accuracy at different threshold
          2) 3) a list of mean and std for error in predicted time to sepsis at different threshold
                    (given that this patient having at least one predicted label 1)
          4)a list of confusion matrices at different threshold
          5)a list of the recognised ratio of septic patients in different bin of time to sepsis, namely, [>0,>6,>18], at different threshold
    """

    CMs = []
    patient_pred_label_list = []
    pred_septic_time_list = []
    for thred in thresholds:
        pred_labels = (prob_preds >= thred).astype('int')

        _, patient_pred_label, CM, pred_septic_time, _, _, _ = patient_level_pred(df, labels_true, pred_labels, a1,
                                                                                  sample_ids)

        CMs.append(CM)
        patient_pred_label_list.append(patient_pred_label)
        pred_septic_time_list.append(pred_septic_time)

    return CMs, patient_pred_label_list, pred_septic_time_list

def suboptimal_choice_patient(labels_true, prob_preds, val_full_indices, \
                              a1=6, n=10, precision=100, discard=-1):
    """

        Finding suboptimal solution by through different threshold for probability on TEST set

        Inputs:
           prob_preds: predicted risk scores concatenated val sets
           labels_true: true corresponding labels
           val_full_indices:  [[val indices fold1],..,[val indices fold x],..]
                in each fold,
                    [val indices fold x]=[[indxs for patient 1 in fold x],..[indxs for patient j in fold x],....]


        Outputs:

          1) a list of confusion matrices at different threshold
          2) For the recognised septic patients, the time in advance we can predict sepsis at different threshold
          3) predicted labels at different threshold on concatenated val sets(i.e., shuffled train set)


    """

    ## setting a sequence of thresholds

    thresholds = np.arange(precision * n)[:discard * n] / precision / n

    CMs = []
    time_difference_list = []
    preds_septic_perpatient_list = []

    for thred in thresholds:
        val_preds = (prob_preds >= thred).astype('int')

        CM, sepsis_time_difference, preds_septic_perpatient = patient_level_performance(val_preds, \
                                                                                        labels_true, \
                                                                                        val_full_indices, \
                                                                                        a1=a1)
        CMs.append(CM)
        time_difference_list.append(sepsis_time_difference)
        preds_septic_perpatient_list.append(preds_septic_perpatient)

    return CMs, time_difference_list, preds_septic_perpatient_list


################## Those for test set #############################


def patient_idx(icustay_lengths):
    """
        idxs for each patient, [[idx for patient 1],..[idx for patient i],...]

    """

    icustay_lengths_cumsum = np.insert(np.cumsum(np.array(icustay_lengths)), 0, 0)
    total_indices = len(icustay_lengths)
    icustay_fullindices = [np.arange(icustay_lengths[i]) + icustay_lengths_cumsum[i] for i in range(total_indices)]

    return icustay_fullindices


def patient_level_test_performance(test_preds, labels_true, test_full_indices, icuid_seq=None, a1=6):
    """

        Mainly computing patient_level performance on test set

        Inputs:
           test_preds: predicted labels on vals
           labels_true: true corresponding labels
           test_full_indices:  [[idxs for patient 1],..[idxs for patient j],....]

        Outputs:

          1) confusion matrix
          2) For the recognised septic patients, the time in advance we can predict sepsis
          3) predicted labels on val sets

    """

    true_septic_perpatient = np.empty((0, 1), int)
    preds_septic_perpatient = np.empty((0, 1), int)

    true_septic_time_perpatient = np.empty((0, 1), int)
    preds_septic_time_perpatient = np.empty((0, 1), int)

    instance_length = np.concatenate(test_full_indices).shape[0]
    patient_length = len(test_full_indices)

    icu_lengths = np.array([len(test_full_indices[i]) for i in range(patient_length)])

    preds_septic_perpatient = np.array(
        [int(len(np.where(test_preds[test_full_indices[j]] > 0)[0]) > 0) for j in range(patient_length)])

    true_septic_perpatient = np.array(
        [int(len(np.where(labels_true[test_full_indices[j]] > 0)[0]) > 0) for j in range(patient_length)])
    preds_septic_time_perpatient = np.array(
        [len(np.where(test_preds[test_full_indices[j]] == 0)[0]) for j in range(patient_length) \
         if true_septic_perpatient[j] != 0 and len(np.where(test_preds[test_full_indices[j]] > 0)[0]) != 0])

    true_septic_time_perpatient = np.array(
        [len(np.where(labels_true[test_full_indices[j]] == 0)[0]) for j in range(patient_length) \
         if true_septic_perpatient[j] != 0 and len(np.where(test_preds[test_full_indices[j]] > 0)[0]) != 0])

    true_septic_time_perpatient = true_septic_time_perpatient + a1

    sepsis_time_difference = true_septic_time_perpatient - preds_septic_time_perpatient

    if icuid_seq is not None:
        icuid_seq_preds_septic = np.array([icuid_seq[j] for j in range(patient_length) \
                                           if true_septic_perpatient[j] != 0 and \
                                           len(np.where(test_preds[test_full_indices[j]] > 0)[0]) != 0])

        return confusion_matrix(true_septic_perpatient, preds_septic_perpatient), \
               sepsis_time_difference, icuid_seq_preds_septic, preds_septic_perpatient
    else:

        return confusion_matrix(true_septic_perpatient, preds_septic_perpatient), \
               sepsis_time_difference, preds_septic_perpatient


def suboptimal_choice_patient_test(labels_true, prob_preds, test_full_indices, \
                                   icuid_seq=None, a1=6, n=10, precision=100, discard=-1):
    """

        Finding suboptimal solution by through different threshold for probability on TEST set

        Inputs:
           prob_preds: predicted risk scores on test set
           labels_true: true corresponding labels
           test_full_indices:  [[idxs for patient 1],..[idxs for patient j],....]

        Outputs:

          1) a list of confusion matrices at different threshold
          2) For the recognised septic patients, the time in advance we can predict sepsis at different threshold
          3) predicted labels at different threshold on test sets


    """
    ## setting a sequence of thresholds

    thresholds = np.arange(precision * n)[:discard * n] / precision / n

    CMs = []
    time_difference_list = []
    icuid_seq_preds_septic_list = []
    preds_septic_perpatient_list = []

    for thred in thresholds:

        test_preds = (prob_preds >= thred).astype('int')
        if icuid_seq != None:
            CM, sepsis_time_difference, \
            icuid_seq_preds_septic, preds_septic_perpatient = patient_level_test_performance(test_preds, \
                                                                                             labels_true, \
                                                                                             test_full_indices, \
                                                                                             icuid_seq, \
                                                                                             a1=a1)
        else:
            CM, sepsis_time_difference, \
            preds_septic_perpatient = patient_level_test_performance(test_preds, labels_true, \
                                                                     test_full_indices, \
                                                                     icuid_seq=icuid_seq, a1=a1)

        CMs.append(CM)
        time_difference_list.append(sepsis_time_difference)
        if icuid_seq != None:
            icuid_seq_preds_septic_list.append(icuid_seq_preds_septic)
        preds_septic_perpatient_list.append(preds_septic_perpatient)

    if icuid_seq != None:
        return CMs, time_difference_list, icuid_seq_preds_septic_list, preds_septic_perpatient_list
    else:
        return CMs, time_difference_list, preds_septic_perpatient_list


################## Some useful functions #############################


def decompose_confusion(CM):
    """
        Given 2dim CM, output Sensitivity,Specificity,precision,FNR
    """

    FP = CM[0].sum() - CM[0, 0]
    FN = CM[1].sum() - CM[1, 1]
    TP = CM[1, 1]
    TN = CM[0, 0]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    acc = (TP + TN) / (FP + FN + TP + TN)

    return TPR, TNR, PPV, FNR, acc


def decompose_cms(CMs):
    """
    Given 2dim CM sequence, output the corresponding sequence of Sensitivity,Specificity,FNR,precision
    """
    tprs, tnrs, pres, fnrs, accs = [], [], [], [], []
    for i in range(len(CMs)):
        tpr, tnr, pre, fnr, acc = decompose_confusion(CMs[i])

        tprs.append(tpr)
        tnrs.append(tnr)
        fnrs.append(fnr)
        pres.append(pre)
        accs.append(acc)

    return np.array(tprs), np.array(tnrs), np.array(fnrs), np.array(pres), np.array(accs)


def patient_level_auc(labels_true, probs_preds, full_indices, precision, n=10, discard=-1, a1=6):
    CMs = []

    thresholds = np.arange(n * precision)[:discard * n] / precision / n

    for thresh in thresholds:
        preds = (probs_preds >= thresh).astype('int')

        CM, _, _ = patient_level_test_performance(preds, labels_true, full_indices, a1=a1)

        CMs.append(CM)

    tprs, tnrs, _, _, _ = decompose_cms(CMs)

    return np.array(tprs), 1 - np.array(tnrs)


def output_at_metric_level_using_CM(CMs, somelist, metric_required=[0.80, 0.85], metric='sensitivity'):
    """
        Given a seq of CMs and corresponding target list, find the target variable at required metric level

    """
    output = []
    metric_now = []
    CMs_output = []

    for i in range(len(CMs)):

        tpr, tnr, ppv, _, _ = decompose_confusion(CMs[i])

        if metric == 'sensitivity':
            metric_now.append(tpr)
        elif metric == 'specificity':
            metric_now.append(tnr)
        elif metric == 'precision':
            metric_now.append(ppv)

    metric_now = np.array(metric_now)

    for j in range(len(metric_required)):
        metric_thred = metric_required[j]

        diff = metric_now - metric_thred
        min_value = np.min(diff[np.where(diff >= 0)[0]])
        idx = np.where(diff == min_value)[0][0]

        output.append(somelist[idx])

        CMs_output.append(CMs[idx])

    return output, CMs_output


def output_at_metric_level(somelist, metric_data, metric_required=[0.80, 0.85]):
    """
        Given a seq of CMs and corresponding target list, find the target variable at required metric level

    """
    metric_now = np.array(metric_data)

    if len(metric_required) == 1:

        metric_thred = metric_required[0]
        diff = metric_now - metric_thred
        min_value = np.min(diff[np.where(diff >= 0)[0]])
        idx = np.where(diff == min_value)[0][0]

        return somelist[idx]

    else:

        output = []
        for j in range(len(metric_required)):
            metric_thred = metric_required[j]

            diff = metric_now - metric_thred
            min_value = np.min(diff[np.where(diff >= 0)[0]])
            idx = np.where(diff == min_value)[0][0]
            output.append(somelist[idx])

        return output


def patient_level_main_outputs_threemodels(labels_list_list, probs_list_list, \
                                           test_indices_list_list, \
                                           icuid_sequence_list_list=None, \
                                           models=['lgbm', 'lstm', 'coxph'], \
                                           definitions=['t_sofa', 't_suspicion', 't_sepsis_min'], \
                                           n=10, a1=6, precision=100):
    """


        outputing main outputs at patient level for three models


        main outputs include:


            tpr, fpr, fnr, precision, time_difference for prediction


        each kind output is in list format as

            [[output list for lgbm],[output list for lstm],[output list for coxph]]

            within [output list for lgbm], the output list contains metrics for three defs:

            [output list for lgbm]=[[output for t_sofa under model lgbm],
                                    [output for t_suspicion under model lgbm],
                                    [output for t_sepsis_min under model lgbm]]



    """

    tprs_list_list, fprs_list_list, fnrs_list_list, \
    pres_list_list, accs_list_list, \
    time_list_list, icuid_seq_list_list = [], [], [], [], [], [], []

    for model in range(len(models)):

        tprs_list, fprs_list, fnrs_list, pres_list, accs_list, time_list, icuid_seq_list = [], [], [], [], [], [], []

        for defi in range(len(definitions)):

            if icuid_sequence_list_list != None:
                CMs, time_differences, icuid_seq, _ = suboptimal_choice_patient_test(labels_list_list[model][defi], \
                                                                                     probs_list_list[model][defi], \
                                                                                     test_indices_list_list[model][
                                                                                         defi], \
                                                                                     icuid_sequence_list_list[model][
                                                                                         defi], \
                                                                                     precision=precision, n=n, a1=a1)
            else:
                CMs, time_differences, _ = suboptimal_choice_patient_test(labels_list_list[model][defi], \
                                                                          probs_list_list[model][defi], \
                                                                          test_indices_list_list[model][defi], \
                                                                          icuid_seq=None, \
                                                                          precision=precision, n=n, a1=a1)

            tpr, tnr, fnr, pre, acc = decompose_cms(CMs)

            tprs_list.append(tpr)
            fprs_list.append(1 - tnr)
            fnrs_list.append(fnr)
            pres_list.append(pre)
            accs_list.append(acc)
            time_list.append(time_differences)

        tprs_list_list.append(tprs_list)
        fprs_list_list.append(fprs_list)
        fnrs_list_list.append(fnrs_list)
        pres_list_list.append(pres_list)
        accs_list_list.append(accs_list)
        time_list_list.append(time_list)

        if icuid_sequence_list_list != None:
            icuid_seq_list_list.append(icuid_seq)
    if icuid_sequence_list_list != None:
        return tprs_list_list, fprs_list_list, fnrs_list_list, pres_list_list, accs_list_list, time_list_list, icuid_seq_list_list
    else:
        return tprs_list_list, fprs_list_list, fnrs_list_list, pres_list_list, accs_list_list, time_list_list


def patient_level_threded_output_threemodels(some_list_list, metric_seq_list_list, \
                                             models=['lgbm', 'lstm', 'coxph'], \
                                             definitions=['t_sofa', 't_suspicion', 't_sepsis_min'], \
                                             metric_required=[0.85]):
    """


        outputing output threding at some metric level of patient level for three models



        The output is in list format as

            [[output list for lgbm],[output list for lstm],[output list for coxph]]

            within [output list for lgbm], the output list contains metrics for three defs:

            [output list for lgbm]=[[output for t_sofa under model lgbm],
                                    [output for t_suspicion under model lgbm],
                                    [output for t_sepsis_min under model lgbm]]



    """

    output_list_list = []

    for model in range(len(models)):

        output_list = []

        for defi in range(len(definitions)):
            output_current = output_at_metric_level(some_list_list[model][defi], \
                                                    metric_seq_list_list[model][defi], \
                                                    metric_required=metric_required)
            output_list.append(output_current)

        output_list_list.append(output_list)

    return output_list_list




