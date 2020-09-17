from src.features.sepsis_mimic3_myfunction import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



def probs_extraction(prob_preds, labels_true, test_full_indices, \
                     a1=4, bins=np.append(np.arange(9) * 6, 1000)):
    """

        Output



          1) predicted probability sequence, namely, 'prob_preds_list'
          2) true labels per patient, namely, 'labels_list'

          on the validation sets in the form of

           [[XXX for patient 1 in validation set],[XXX for patient 2 in validation set],[],...]


          Also output
          3)sepsis label for patients, namely, 'true_septic_perpatient'
          4)length of icu stay for patients, namely, 'length_perpatien'

          Finally, for the purpose of creating plots in James paper
          (Figure 2, "Proportion of sepsis cases predicted correctly in different time windows
                        tuned to different precision levels") I output

          5)the number of septic patients in different bin of time to sepsis, namely, "septic_length_hist"



    """

    prob_preds_list = []
    labels_list = []

    true_septic_perpatient = np.empty((0, 1), int)
    length_perpatient = np.empty((0, 1), int)
    true_septic_lengths = np.empty((0, 1), int)

    tt = 0
    k = len(test_full_indices)

    for i in range(k):
        current_length = np.concatenate(test_full_indices[i]).shape[0]
        current_patient_num = len(test_full_indices[i])

        current_length_perpatient = np.array([len(test_full_indices[i][j]) for j in range(current_patient_num)])
        length_perpatient = np.append(length_perpatient, current_length_perpatient)

        current_length_cumsum = np.insert(np.cumsum(np.array(current_length_perpatient)), 0, 0)
        current_fullindices = [int(tt) + np.arange(current_length_perpatient[j]) + current_length_cumsum[j] for j in
                               range(current_patient_num)]

        tt += current_length

        prob_preds_list_current = [prob_preds[current_fullindices[j]] for j in range(current_patient_num)]
        prob_preds_list = prob_preds_list + prob_preds_list_current

        labels_list_current = [labels_true[current_fullindices[j]] for j in range(current_patient_num)]
        labels_list = labels_list + labels_list_current

        current_true_septic_perpatient = np.array(
            [int(len(np.where(labels_true[current_fullindices[j]] > 0)[0]) > 0) for j in range(current_patient_num)])
        true_septic_perpatient = np.append(true_septic_perpatient, current_true_septic_perpatient)

        current_true_septic_lengths = np.array(
            [len(np.where(labels_true[current_fullindices[j]] == 0)[0]) + a1 for j in range(current_patient_num) if
             len(np.where(labels_true[current_fullindices[j]] > 0)[0]) > 0])
        true_septic_lengths = np.append(true_septic_lengths, \
                                        current_true_septic_lengths)

    septic_length_hist = np.histogram(true_septic_lengths, bins=bins)[0]

    return prob_preds_list, labels_list, true_septic_perpatient, length_perpatient, septic_length_hist


def patient_level_performance(train_preds, labels_true, test_full_indices, number_sepsis_hist, \
                              a1=4, bins=np.append(np.arange(9) * 6, 1000), sampling=False):
    """

        Mainly computing patient_level performance

        Outputs:

          1) accuracy
          2) 3) mean and std for error in predicted time to sepsis
                    (given that this patient having at least one predicted label 1)

          4)confusion matrix

          5)the recognised ratio of septic patients in different bin of time to sepsis, namely, [>0,>6,>18]

    """

    true_septic_perpatient = np.empty((0, 1), int)
    preds_septic_perpatient = np.empty((0, 1), int)

    true_septic_time_perpatient = np.empty((0, 1), int)
    preds_septic_time_perpatient = np.empty((0, 1), int)

    #     uncaptured_septic_time=np.empty((0,1),int)

    tt = 0
    k = len(test_full_indices)

    for i in range(k):

        if sampling:
            print('Fold', i)

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

        #         current_preds_septic_lengths_perpatient=np.array([np.where(labels_true[current_fullindices[j]]>0)[0][0] for j in range(current_patient_num) \
        #                                                       if current_true_septic_perpatient[j]!=0 and\
        #                                                       len(np.where(train_preds[current_fullindices[j]]>0)[0])!=0])

        #         current_true_septic_time_perpatient_unpredictable=np.array([np.where(labels_true[current_fullindices[j]]>0)[0][0] for j in range(current_patient_num) \
        #                                                       if current_true_septic_perpatient[j]!=0 and\
        #                                                       len(np.where(train_preds[current_fullindices[j]]>0)[0])==0])

        #         uncaptured_septic_time=np.append(uncaptured_septic_time,current_true_septic_time_perpatient_unpredictable)

        if sampling:
            index_sample = np.where(current_true_septic_perpatient == 1)[0][-1]
            print('sample with true:', labels_true[current_fullindices[index_sample]])
            print('preds:', train_preds[current_fullindices[index_sample]])
            print('\n')

    true_septic_time_perpatient = true_septic_time_perpatient + a1

    ### Those to creat comprable figure in James' paper
    preds_septic_lengths = preds_septic_time_perpatient + a1
    preds_septic_lengths_hist = np.histogram(preds_septic_lengths, \
                                             bins=bins)[0]
    preds_sepsis_props = preds_septic_lengths_hist / number_sepsis_hist

    sepsis_time_difference = np.abs(true_septic_time_perpatient - preds_septic_time_perpatient)

    return accuracy_score(true_septic_perpatient, preds_septic_perpatient), \
           np.mean(sepsis_time_difference), \
           np.std(sepsis_time_difference), \
           confusion_matrix(true_septic_perpatient, preds_septic_perpatient), \
           preds_sepsis_props


def suboptimal_choice_patient(labels_true, prob_preds, \
                              train_preds, \
                              test_full_indices, \
                              septic_lengths_hist, \
                              bins=np.append(np.arange(9) * 6, 1000), \
                              thresholds=np.arange(100)[1:-20] / 100, \
                              sampling=False):
    """

        Finding suboptimal solution by through different threshold for probability


        Outputs:

          1) a list of accuracy at different threshold
          2) 3) a list of mean and std for error in predicted time to sepsis at different threshold
                    (given that this patient having at least one predicted label 1)

          4)a list of confusion matrices at different threshold

          5)a list of the recognised ratio of septic patients in different bin of time to sepsis, namely, [>0,>6,>18], at different threshold

    """

    mean_septic_times = []
    accuracys = []
    std_septic_times = []
    CMs = []
    lengths_props = []

    for thred in thresholds:
        #        print('Thresholding at', thred)

        #         fpr, tpr, thresholds = roc_curve(labels_true, prob_preds, pos_label=1)
        #         index=np.where(tpr>=thred)[0][0]
        train_preds = (prob_preds >= thred).astype('int')
        #         print('auc and sepcificity',roc_auc_score(labels_true,prob_preds),1-fpr[index])
        #         print('accuracy',accuracy_score(labels_true,train_preds))

        accuracy, mean_septic_time_error, std_septic_time_error, CM, septic_length_proportion = patient_level_performance(
            train_preds, \
            labels_true, \
            test_full_indices, \
            septic_lengths_hist, \
            bins=bins, \
            sampling=sampling)
        accuracys.append(accuracy)
        mean_septic_times.append(mean_septic_time_error)
        std_septic_times.append(std_septic_time_error)
        CMs.append(CM)
        lengths_props.append(septic_length_proportion)

    return np.array(accuracys), np.array(mean_septic_times), np.array(std_septic_times), CMs, lengths_props


def proportion_output_precision(CMs, septic_length_hist, thresholds, prvs=[0.25, 0.375, 0.5]):
    proportions = []
    prvs_now = []
    CMs_now = []
    for i in range(len(CMs)):
        tpr, tnr, prv = decompose_confusion(CMs[i])
        prvs_now.append(prv)

    prvs_now = np.array(prvs_now)

    for j in range(len(prvs)):
        prv_thred = prvs[j]

        idx = np.where(prvs_now >= prv_thred)[0][0]

        proportions.append(septic_length_hist[idx])
        CMs_now.append(CMs[idx])

    return np.asarray(proportions), CMs_now


def decompose_confusion(CM):
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

    return TPR, TNR, PPV


def trajectory_plot(probs_sample, labels_sample, \
                    labels=['Risk score', 'Labels for T=6', 'Ground truth'], \
                    figsize=(10, 3), fontsize=14, savename=None):
    plt.figure(figsize=figsize)

    plt.plot(probs_sample, linestyle='-.', lw=2, label='Risk score')

    new_time_seq, new_path, true_times, true_labels = rect_line_at_turn(labels_sample)

    plt.plot(new_time_seq, new_path, lw=2, label='Labels for T=6')

    plt.plot(true_times, true_labels, linestyle=':', label='Ground truth', lw=2)

    plt.xlabel('ICU length-of-stay since ICU admission (Hour) of one spetic patient', fontsize=fontsize)

    plt.legend(loc='upper left', bbox_to_anchor=(1.005, 1), fontsize=fontsize - 1)

    plt.xticks(fontsize=fontsize - 1)
    plt.yticks(fontsize=fontsize - 1)

    if savename is not None:

        plt.savefig(savename, dpi=300, bbox_inches='tight')
    else:
        plt.show()