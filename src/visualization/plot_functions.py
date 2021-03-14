import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from matplotlib_venn import venn3
from matplotlib.patches import Rectangle

import sys

sys.path.insert(0, '../')

import constants

import src.features.sepsis_mimic3_myfunction as mimic3_myfunc
import src.visualization.sepsis_mimic3_myfunction_patientlevel as mimic3_myfunc_patientlevel
from src.visualization.sepsis_mimic3_myfunction_patientlevel import decompose_confusion

colors_barplot = sns.color_palette()
colors_auc = sns.color_palette("Dark2")
linestyles = [':', '-.', '-', '--']


############################ For auc plots ############################

def auc_plot(trues_list, probs_list, names, fontsize=14, \
             colors=colors_auc, linestyles=linestyles, \
             lw=2, loc="lower right", save_name=None):
    """
        AUC plots in one figure via ground truth and predicted probabilities

    Input:

        trues_list: ground-truth-seq list

                eg, for 2 set of data, [[ground truth for set1],[ground truth for set2]]

        probs_list: probability-seq list

            eg, for 2 set of data, [[probabilities for set1],[probabilities for set2]]

        names: curve labels

        save_name: if None: print figure; else: save to save_name.png

    """

    num = len(trues_list)

    plt.figure()

    for i in range(num):
        fpr, tpr, _ = roc_curve(trues_list[i], probs_list[i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i], linestyle=linestyles[i], \
                 lw=lw, label='ROC curve for ' + names[i] + ' (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.legend(loc=loc, fontsize=fontsize - 3)

    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    if save_name is not None:
        plt.savefig(save_name + '.jpeg', dpi=350)
    else:
        plt.show()


def auc_plot_xy_pairs(model=constants.MODELS[0], current_data='blood_culture_data/', \
                      precision=100, n=100, a1=6, names=['48,24', '24,12', '12,6', '6,3'], purpose='test'):
    """
        For each definition and fixed model, producing two AUC plots, one online prediction,one patien-level, across 4 different xy_pairs

    """

    Root_Data, _, _, Output_predictions, Output_results = mimic3_myfunc.folders(current_data, model=model)

    for definition in constants.FEATURES:

        labels_list = []
        probs_list = []
        tprs_list = []
        fprs_list = []

        for x, y in constants.xy_pairs:
            print(definition, x, y, model)
            Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/' + purpose + '/'

            labels_now = np.load(
                Data_Dir + 'label' + definition[1:] + '_' + str(x) + '_' + str(y) + '_' + str(a1) + '.npy')

            probs_now = np.load(Output_predictions + purpose + '/prob_preds_' + str(x) + '_' + str(y) + '_' + str(
                a1) + '_' + definition[1:] + '.npy')

            icu_lengths_now = np.load(
                Data_Dir + 'icustay_lengths' + definition[1:] + '_' + str(x) + '_' + str(y) + '.npy')

            icustay_fullindices_now = mimic3_myfunc_patientlevel.patient_idx(icu_lengths_now)

            tpr, fpr = mimic3_myfunc_patientlevel.patient_level_auc(labels_now, probs_now, icustay_fullindices_now,
                                                                    precision, n=n, a1=a1)

            labels_list.append(labels_now)
            probs_list.append(probs_now)
            tprs_list.append(tpr)
            fprs_list.append(fpr)

        auc_plot(labels_list, probs_list, names=names, \
                 save_name=Output_results + 'auc_plot_instance_level_' + model + definition[1:] + '_' + purpose)
        auc_plot_patient_level(fprs_list, tprs_list, names=names, \
                               save_name=Output_results + 'auc_plot_patient_level_' + model + definition[
                                                                                              1:] + '_' + purpose)

    #########################For CI ################################################


from scipy.stats import norm

n_bootstraps = 100
# rng_seed = 1  # control reproducibility
alpha = 0.95


def CI_AUC_bootstrapping(n_bootstraps, alpha, y_true, y_pred, rng_seed=1):
    # to compute alpha % confidence interval using boostraps for n_boostraps times
    bootstrapped_scores = []
    fprs, tprs = [], []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        # sample_index = np.random.choice(range(0, len(y_pred)), len(y_pred))
        # print(indices)

        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        fpr, tpr, _ = roc_curve(y_true[indices], y_pred[indices])
        fprs.append(fpr)
        tprs.append(tpr)
        bootstrapped_scores.append(score)
    #         if i%20 ==0:
    #             print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    factor = norm.ppf(alpha)
    std1 = np.std(bootstrapped_scores)
    mean1 = np.mean(bootstrapped_scores)
    up1 = mean1 + factor * std1
    lower1 = mean1 - factor * std1
    #     print( '{}% confidence interval is [{},{}]'.format(alpha, up1, lower1))
    return [lower1, up1], fprs, tprs


def fprs_tprs_output(labels_list_list, probs_list_list, n_bootstraps=100, alpha=0.95):
    fprs_lists = [[] for kk in range(len(constants.MODELS))]
    tprs_lists = [[] for kk in range(len(constants.MODELS))]

    for i in range(len(constants.MODELS)):

        fprs_lists[i] = [[] for k in range(len(constants.FEATURES))]
        tprs_lists[i] = [[] for k in range(len(constants.FEATURES))]

        for j in range(len(constants.FEATURES)):
            CI_results, fprs, tprs = CI_AUC_bootstrapping(n_bootstraps, alpha, labels_list_list[i][j], \
                                                          probs_list_list[i][j], rng_seed=1)

            print(constants.MODELS[i], constants.FEATURES[j], "{:.3f}".format(roc_auc_score(labels_list_list[i][j], \
                                                                                            probs_list_list[i][j])), \
                  "[" + "{:.3f}".format(CI_results[0]) + "," + "{:.3f}".format(CI_results[1]) + "]")

            fprs_lists[i][j] += fprs
            tprs_lists[i][j] += tprs

        print('\n')

    return fprs_lists, tprs_lists


def CI_std_output(fprs_lists, tprs_lists, \
                  mean_fpr_list=[np.linspace(0, 1, 30 + 0 * i) for i in range(3)]):
    error_list = [[] for i in range(len(constants.MODELS))]

    for i in range(len(constants.MODELS)):

        error_list[i] = [[] for k in range(len(constants.FEATURES))]

        for j in range(len(constants.FEATURES)):
            tprs_ = []

            for k in range(len(tprs_lists[i][j])):
                fpr_now = fprs_lists[i][j][k]
                tpr_now = tprs_lists[i][j][k]
                interp_tpr = np.interp(mean_fpr_list[i], fpr_now, tpr_now)
                interp_tpr[0] = 0.0
                tprs_.append(interp_tpr)

            mean_tpr = np.mean(tprs_, axis=0)
            mean_tpr[-1] = 1.0

            std_tpr = np.std(tprs_, axis=0)
            error_list[i][j] = std_tpr

    return error_list


# colors_shade=sns.color_palette("Set2")

colors_shade = sns.color_palette("Pastel2")


def auc_subplots(trues_list, probs_list, error_lists, names, \
                 mean_fpr_list=[np.linspace(0, 1, 30 + 0 * i) for i in range(3)], \
                 fontsize=14, figsize=(15, 5), titles=constants.MODELS, \
                 colors=[colors_shade[0], colors_shade[2], colors_shade[1]], \
                 colors_line=[colors_auc[0], colors_auc[2], colors_auc[1]], \
                 linestyles=linestyles, lw=2, loc="lower right", save_name=None):
    """

        AUC plots for different models via ground truth and predicted probabilities

    Input:

        trues_list: list of ground-truth-seq lists

                eg, for three models for 2 set of data,[[model1 truth-list], [model2 truth-list], [model3 truth-list]]
                    [model1 truth-list]=[[ground truth for set1],[ground truth for set2]]

        probs_list: probability-seq list

            eg, for three models for 2 set of data,  [[model1 probs-list], [model2 probs-list], [model3 probs-list]]

                [model1 probs-list]=[[probabilities for set1],[probabilities for set2]]

        names: curve labels for sets of data

        save_name: if None: print figure; else: save to save_name.png



    """

    plt.figure(figsize=figsize)
    plt.subplot(131)

    num = len(trues_list[0])

    for i in range(num):
        fpr, tpr, _ = roc_curve(trues_list[0][i], probs_list[0][i])
        roc_auc = auc(fpr, tpr)

        mean_tpr = np.interp(mean_fpr_list[i], fpr, tpr)
        plt.plot(mean_fpr_list[i], mean_tpr, color=colors_line[i], \
                 lw=lw, linestyle=linestyles[i], label='ROC curve for ' + names[i] + ' (area = %0.2f)' % roc_auc)

        #         plt.errorbar(mean_fpr_list[i], mean_tpr, error_lists[-1][i], color=colors[i],\
        #                     lw=lw,linestyle=linestyles[i],label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        plt.fill_between(mean_fpr_list[i], mean_tpr - error_lists[-1][i], mean_tpr + error_lists[-1][i],
                         color=colors[i])

    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.title(titles[0])
    plt.legend(loc=loc, fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    plt.subplot(132)

    num = len(trues_list[1])
    for i in range(num):
        fpr, tpr, _ = roc_curve(trues_list[1][i], probs_list[1][i])
        roc_auc = auc(fpr, tpr)

        mean_tpr = np.interp(mean_fpr_list[i], fpr, tpr)

        plt.plot(mean_fpr_list[i], mean_tpr, color=colors_line[i], \
                 lw=lw, linestyle=linestyles[i], label='ROC curve for ' + names[i] + ' (area = %0.2f)' % roc_auc)

        #         plt.errorbar(mean_fpr_list[i], mean_tpr, error_lists[-1][i], color=colors[i],\
        #                     lw=lw,linestyle=linestyles[i],label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        plt.fill_between(mean_fpr_list[i], mean_tpr - error_lists[-1][i], mean_tpr + error_lists[-1][i],
                         color=colors[i])

    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.title(titles[1])
    plt.legend(loc=loc, fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    plt.subplot(133)

    num = len(trues_list[-1])
    for i in range(num):
        fpr, tpr, _ = roc_curve(trues_list[-1][i], probs_list[-1][i])
        roc_auc = auc(fpr, tpr)

        mean_tpr = np.interp(mean_fpr_list[i], fpr, tpr)
        plt.plot(mean_fpr_list[i], mean_tpr, color=colors_line[i], \
                 lw=lw, linestyle=linestyles[i], label='ROC curve for ' + names[i] + ' (area = %0.2f)' % roc_auc)

        #         plt.errorbar(mean_fpr_list[i], mean_tpr, error_lists[-1][i], color=colors[i],\
        #                     lw=lw,linestyle=linestyles[i],label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        plt.fill_between(mean_fpr_list[i], mean_tpr - error_lists[-1][i], mean_tpr + error_lists[-1][i],
                         color=colors[i])

    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.title(titles[-1])
    plt.legend(loc=loc, fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    if save_name is not None:
        plt.savefig(save_name + '.jpeg', dpi=350)
    else:

        plt.show()

        ############## Patient level ####################


def patient_level_probability_max(probs):
    return np.max(probs)


def fprs_tprs_output_patient_level(labels_list_list, probs_list_list, indices_list_list, n_bootstraps=100, alpha=0.95):
    fprs_lists = [[] for kk in range(len(constants.MODELS))]
    tprs_lists = [[] for kk in range(len(constants.MODELS))]

    probs_list = [[] for kk in range(len(constants.MODELS))]
    labels_list = [[] for kk in range(len(constants.MODELS))]

    for i in range(len(constants.MODELS)):

        fprs_lists[i] = [[] for k in range(len(constants.FEATURES))]
        tprs_lists[i] = [[] for k in range(len(constants.FEATURES))]

        probs_list[i] = [[] for k in range(len(constants.FEATURES))]
        labels_list[i] = [[] for k in range(len(constants.FEATURES))]

        for j in range(len(constants.FEATURES)):
            full_idxs = indices_list_list[i][j]
            par_probs = [patient_level_probability_max(probs_list_list[i][j][full_idxs[k]]) for k in
                         range(len(full_idxs))]

            par_labels = [labels_list_list[i][j][full_idxs[k]][-1] for k in range(len(full_idxs))]

            CI_results, fprs, tprs = CI_AUC_bootstrapping(n_bootstraps, alpha, np.array(par_labels),
                                                          np.array(par_probs), rng_seed=1)

            print(constants.MODELS[i], constants.FEATURES[j], \
                  "{:.3f}".format(roc_auc_score(par_labels, par_probs)), \
                  "[" + "{:.3f}".format(CI_results[0]) + "," + "{:.3f}".format(CI_results[1]) + "]")

            fprs_lists[i][j] += fprs
            tprs_lists[i][j] += tprs

            probs_list[i][j] = np.array(par_probs)
            labels_list[i][j] = np.array(par_labels)

        print('\n')

    return fprs_lists, tprs_lists, labels_list, probs_list


colors_shade = [sns.color_palette("Pastel2")[0], sns.color_palette("Pastel2")[2], sns.color_palette("Pastel2")[1]]
colors_auc = sns.color_palette("Dark2")
colors_auc = [colors_auc[0], colors_auc[2], colors_auc[1]]


def auc_subplots_errorbars(trues_list, probs_list, error_lists, names, \
                           mean_fpr_list=[np.linspace(0, 1, 30 + 0 * i) for i in range(3)], \
                           fontsize=14, figsize=(15, 5), titles=constants.MODELS, \
                           colors=colors_shade, colors_line=colors_auc, linestyles=linestyles, lw=2, \
                           loc="lower right", save_name=None):
    """
        
        AUC plots for different models via ground truth and predicted probabilities
        
    Input:
    
        trues_list: list of ground-truth-seq lists 
        
                eg, for three models for 2 set of data,[[model1 truth-list], [model2 truth-list], [model3 truth-list]]
                    [model1 truth-list]=[[ground truth for set1],[ground truth for set2]]
                
        probs_list: probability-seq list
        
            eg, for three models for 2 set of data,  [[model1 probs-list], [model2 probs-list], [model3 probs-list]]
            
                [model1 probs-list]=[[probabilities for set1],[probabilities for set2]]
            
        names: curve labels for sets of data
        
        save_name: if None: print figure; else: save to save_name.png

        
    
    """

    plt.figure(figsize=figsize)
    plt.subplot(131)

    num = len(trues_list[0])

    for i in range(num):
        fpr, tpr, _ = roc_curve(trues_list[0][i], probs_list[0][i])
        roc_auc = auc(fpr, tpr)

        mean_tpr = np.interp(mean_fpr_list[i], fpr, tpr)
        plt.plot(mean_fpr_list[i], mean_tpr, color=colors_line[i], \
                 lw=lw, linestyle=linestyles[i], label='ROC curve for ' + names[i] + ' (area = %0.2f)' % roc_auc)

        #         plt.errorbar(mean_fpr_list[i], mean_tpr, error_lists[-1][i], color=colors[i],\
        #                     lw=lw,linestyle=linestyles[i],label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        plt.fill_between(mean_fpr_list[i], mean_tpr - error_lists[-1][i], mean_tpr + error_lists[-1][i],
                         color=colors[i])

    #         plt.plot(fpr, tpr,  color=colors_line[i], lw=lw,linestyle=linestyles[i],\
    #                  label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)

    #         error=np.interp(fpr,mean_fpr_list[i], error_lists[0][i])
    #         plt.fill_between(fpr, tpr-error, tpr+error, color=colors[i])

    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.title(titles[0])
    plt.legend(loc=loc, fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    plt.subplot(132)

    num = len(trues_list[1])
    for i in range(num):
        fpr, tpr, _ = roc_curve(trues_list[1][i], probs_list[1][i])
        roc_auc = auc(fpr, tpr)

        mean_tpr = np.interp(mean_fpr_list[i], fpr, tpr)

        plt.plot(mean_fpr_list[i], mean_tpr, color=colors_line[i], \
                 lw=lw, linestyle=linestyles[i], label='ROC curve for ' + names[i] + ' (area = %0.2f)' % roc_auc)

        #         plt.errorbar(mean_fpr_list[i], mean_tpr, error_lists[-1][i], color=colors[i],\
        #                     lw=lw,linestyle=linestyles[i],label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        plt.fill_between(mean_fpr_list[i], mean_tpr - error_lists[-1][i], mean_tpr + error_lists[-1][i],
                         color=colors[i])

    #         plt.plot(fpr, tpr,  color=colors_line[i], lw=lw,linestyle=linestyles[i],\
    #                  label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)

    #         error=np.interp(fpr,mean_fpr_list[i], error_lists[1][i])
    #         plt.fill_between(fpr, tpr-error, tpr+error, color=colors[i])

    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.title(titles[1])
    plt.legend(loc=loc, fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    plt.subplot(133)

    num = len(trues_list[-1])
    for i in range(num):
        fpr, tpr, _ = roc_curve(trues_list[-1][i], probs_list[-1][i])
        roc_auc = auc(fpr, tpr)

        mean_tpr = np.interp(mean_fpr_list[i], fpr, tpr)
        plt.plot(mean_fpr_list[i], mean_tpr, color=colors_line[i], \
                 #                     lw=lw,linestyle=linestyles[i],label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
                 lw=lw, linestyle=linestyles[i], label='ROC curve for ' +names[i] + ' (area = %0.2f)' % roc_auc)

        #         plt.errorbar(mean_fpr_list[i], mean_tpr, error_lists[-1][i], color=colors[i],\
        #                     lw=lw,linestyle=linestyles[i],label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        plt.fill_between(mean_fpr_list[i], mean_tpr - error_lists[-1][i], mean_tpr + error_lists[-1][i],
                         color=colors[i])

    #         plt.plot(fpr, tpr,  color=colors_line[i], lw=lw,linestyle=linestyles[i],\
    #                  label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)

    #         error=np.interp(fpr,mean_fpr_list[i], error_lists[-1][i])
    #         plt.fill_between(fpr, tpr-error, tpr+error, color=colors[i])

    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.title(titles[-1])
    plt.legend(loc=loc, fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    if save_name is not None:
        plt.savefig(save_name + '.jpeg', dpi=350)
    else:

        plt.show()


def recall_specificity_subplots_patient_level(pres_list, tprs_list, names, \
                                              fontsize=14, figsize=(15, 5), \
                                              titles=constants.MODELS, colors=colors_auc, \
                                              linestyles=linestyles, \
                                              loc="lower left", lw=2, \
                                              save_name=None):
    """



        recall_specificity plots for different models via computed precisions and tprs

    Input:

        pres_list: list of precision lists

                eg, for three models for 2 set of data,[[model1 precision-list], [model2 precision-list], [model3 precision-list]]
                    [model1 precision-list]=[[precision for data set1],[precision for data set2]]

        tprs_list: list of tpr list

                eg, for three models for 2 set of data,[[model1 tpr-list], [model2 tpr-list], [model3 tpr-list]]
                    [model1 tpr-list]=[[tpr for data set1],[tpr for data set2]]


        names: curve labels for sets of data

        save_name: if None: print figure; else: save to save_name.png



    """

    plt.figure(figsize=figsize)
    plt.subplot(131)

    num = len(tprs_list[0])

    for i in range(num):
        plt.plot(pres_list[0][i], tprs_list[0][i], color=colors[i], linestyle=linestyles[i], \
                 lw=lw, label=names[i])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision', fontsize=fontsize)
    plt.ylabel('Recall', fontsize=fontsize)
    plt.title(titles[0])
    plt.legend(loc=loc, fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    plt.subplot(132)

    num = len(tprs_list[1])

    for i in range(num):
        plt.plot(pres_list[1][i], tprs_list[1][i], color=colors[i], linestyle=linestyles[i], \
                 lw=lw, label=names[i])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision', fontsize=fontsize)
    plt.title(titles[1])
    plt.legend(loc=loc, fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    plt.subplot(133)

    num = len(tprs_list[-1])

    for i in range(num):
        plt.plot(pres_list[0][i], tprs_list[0][i], color=colors[i], linestyle=linestyles[i], \
                 lw=lw, label=names[i])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision', fontsize=fontsize)
    plt.title(titles[-1])
    plt.legend(loc=loc, fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    if save_name is not None:
        plt.savefig(save_name + '.jpeg', dpi=350)
    else:

        plt.show()


def auc_plot_patient_level(fprs, tprs, names, fontsize=14, \
                           colors=colors_auc, titles=constants.MODELS, \
                           linestyles=linestyles, lw=2, \
                           loc="lower right", save_name=None):
    """
        AUC plots in one figure via computed fprs and tprs
        
    Input:
    
        fprs: fpr list for different sets of data
        
                eg, for 2 set of data, [[fpr for data set1],[fpr for data set2]]
                
        tprs: tpr list for different sets of data
        
                eg, for 2 set of data, [[tpr for data set1],[tpr for data set2]]

            
        names: curve labels
        
        save_name: if None: print figure; else: save to save_name.png

    
    
    """

    num = len(fprs)
    plt.figure()

    for i in range(num):
        roc_auc = auc(fprs[i], tprs[i])

        plt.plot(fprs[i], tprs[i], color=colors[i], linestyle=linestyles[i], \
                 lw=lw, label='ROC curve for ' + names[i] + ' (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.legend(loc=loc, fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    if save_name is not None:
        plt.savefig(save_name + '.jpeg', dpi=350)
    else:

        plt.show()


############################ For trajectory level plot ############################

def finding_icuid_idx(idx1_septic, test_patient_indices, icustay_lengths=None, \
                      Data_Dir='./Sep_24_12_experiments_new/icustay_id', \
                      definition='t_suspicion'):
    idx1_septic_original = np.concatenate(test_patient_indices)[idx1_septic]

    if icustay_lengths is not None:
        print('The length of current patient', icustay_lengths[idx1_septic_original])

    icuid_sequence = np.load(Data_Dir + definition[1:] + '.npy')

    return icuid_sequence[idx1_septic_original]


def finding_sample_idx(idx1_septicicuid, test_patient_indices, icustay_lengths=None, \
                       Data_Dir='./Sep_24_12_experiments_new/icustay_id', \
                       definition='t_sepsis_min'):
    icuid_sequence = np.load(Data_Dir + definition[1:] + '.npy')

    idx1_septic_original = np.where(icuid_sequence == idx1_septicicuid)[0][0]

    if icustay_lengths is not None:
        print('The length of current patient', icustay_lengths[idx1_septic_original])

    test_patients = np.concatenate(test_patient_indices)

    return np.where(test_patients == idx1_septic_original)[0][0]


def finding_sample_idxs(idx1_septicicuid, test_patient_indices, icustay_lengths=None, \
                        Data_Dir='./Sep_24_12_experiments_new/icustay_id', \
                        definition='t_sepsis_min'):
    icuid_sequence = np.load(Data_Dir + definition[1:] + '.npy')

    idx1_septic_original = np.array([np.where(icuid_sequence == idx1_septicicuid[i])[0][0] \
                                     for i in range(len(idx1_septicicuid))])

    if icustay_lengths is not None:
        print('The length of current patient', icustay_lengths[idx1_septic_original])

    test_patients = np.concatenate(test_patient_indices)

    return np.array([np.where(test_patients == idx1_septic_original[i])[0][0] \
                     for i in range(len(idx1_septic_original))])


def tresholding(probs_sample, thred):
    idx = np.where(probs_sample >= thred)[0][0]
    return np.array([int(i >= idx) for i in range(len(probs_sample))])


def rect_line_at_turn(path, turn_value=1, replace_value=0):
    """
        change binary data such that when there is a turn, we have rectangle turn

        for example, [0,0,1,1] with hidden time [0,1,2,3] will be transformed to [0,0,0,1,1] with time [0,1,2,2,3]
    return:
        new time seq [0,1,2,2,3], new path [0,0,0,1,1], original time seq [0,1,2,3] and old path (the input path)

    """
    num = len(path)

    time_seq = np.arange(num)

    try:
        repeated_idx = np.where(path == turn_value)[0][0]

        new_path = np.insert(path, repeated_idx, 0)
        new_time_seq = np.insert(time_seq, repeated_idx, time_seq[repeated_idx])

        true_labels = np.zeros(len(new_path))
        true_labels[-1] = 1

        true_times = np.arange(len(new_path))
        true_times[-1] = true_times[-2]

        return new_time_seq, new_path, true_times, true_labels
    except:

        return np.arange(len(path)), path, np.arange(len(path)), path


def trajectory_plot(probs_sample, labels_sample, thred=None, \
                    labels=['Risk score', 'Labels for T=6', 'Ground truth'], \
                    figsize=(10, 3), fontsize=14, savename=None):
    plt.figure(figsize=figsize)

    new_time_seq, new_path, true_times, true_labels = rect_line_at_turn(labels_sample)

    plt.plot(true_times, true_labels, linestyle='-.', label='Ground truth', lw=2, color=sns.color_palette()[1])

    plt.plot(new_time_seq, new_path, lw=2, label='Labels for T=6', color=sns.color_palette()[1])

    plt.plot(probs_sample, linestyle=':', lw=1.5, marker='x', label='Risk score', color=sns.color_palette()[0])

    if thred is not None:
        threds = [thred for i in range(len(probs_sample))]

        plt.plot(threds, lw=2, linestyle=':', label='Threshold', color=sns.color_palette()[2])

        pred_labels = tresholding(probs_sample, thred)

        new_time_pred, pred_paths, _, _ = rect_line_at_turn(pred_labels)

        plt.plot(new_time_pred, pred_paths, lw=2, label='Predicted labels', color=sns.color_palette()[2])

    plt.xlabel('ICU length-of-stay since ICU admission (Hour) of one septic patient', fontsize=fontsize)

    plt.legend(loc='upper left', bbox_to_anchor=(1.005, 1), fontsize=fontsize - 1)

    plt.xticks(fontsize=fontsize - 1)
    plt.yticks(fontsize=fontsize - 1)

    if savename is not None:

        plt.savefig(savename + '.jpeg', dpi=350, bbox_inches='tight')
    else:
        plt.show()


#####################sepsis onset time plot functions####################
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


def suboptimal_choice_patient(df, labels_true, prob_preds, a1=6, thresholds=np.arange(100)[1:-20] / 100,
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

        _, patient_pred_label, CM, pred_septic_time, _, _, _ = patient_level_pred(df, labels_true, pred_labels,
                                                                                  a1,
                                                                                  sample_ids)

        CMs.append(CM)
        patient_pred_label_list.append(patient_pred_label)
        pred_septic_time_list.append(pred_septic_time)

    return CMs, patient_pred_label_list, pred_septic_time_list


def venn_3counts(a, b, c):
    ab = np.intersect1d(a, b)
    abc = np.intersect1d(ab, c)

    abc_len = len(abc)
    ab_minusc = len(ab) - abc_len

    bc = np.intersect1d(b, c)
    bc_minusa = len(bc) - abc_len

    ac = np.intersect1d(a, c)
    ac_minusb = len(ac) - abc_len

    solo_a = len(a) - (ab_minusc + abc_len + ac_minusb)
    solo_b = len(b) - (ab_minusc + abc_len + bc_minusa)
    solo_c = len(c) - (bc_minusa + abc_len + ac_minusb)

    return solo_a, solo_b, ab_minusc, solo_c, ac_minusb, bc_minusa, abc_len


def output_metric_level(CMs, pred_sepsispatient_list, levels=[0.88], test_metric='specificity'):
    output = []

    prvs_now = []
    tprs_now = []
    tnrs_now = []

    for i in range(len(CMs)):
        tpr, tnr, prv, _, _ = decompose_confusion(CMs[i])
        prvs_now.append(prv)
        tnrs_now.append(tnr)
        tprs_now.append(tpr)

    prvs_now = np.array(prvs_now)
    tnrs_now = np.array(tnrs_now)
    tprs_now = np.array(tprs_now)

    for j in range(len(levels)):
        metric_thred = levels[j]
        if test_metric == 'precision':
            diff = prvs_now - metric_thred
            min_value = np.min(diff[np.where(diff >= 0)[0]])
            idx = np.where(diff == min_value)[0][0]
        elif test_metric == 'sensitivity':
            diff = tprs_now - metric_thred
            min_value = np.min(diff[np.where(diff >= 0)[0]])
            idx = np.where(diff == min_value)[0][0]

        elif test_metric == 'specificity':
            diff = tnrs_now - metric_thred
            min_value = np.min(diff[np.where(diff >= 0)[0]])
            idx = np.where(diff == min_value)[0][0]

        output.append(pred_sepsispatient_list[idx])

    return output, prvs_now[idx], tprs_now[idx], tnrs_now[idx], idx


def plot_venn(x, y, T, test_metric, metric_thresh, precision, save_dir):
    """

    :param x:
    :param y:
    :param T:
    :param test_metric:
    :param metric_thresh:
    :param precision:
    :param cohort:
    :return:
    """
    definitions = ['t_sofa', 't_suspicion', 't_sepsis_min']
    current_data = 'blood_only_data/'
    Root_Data = constants.DATA_processed + current_data + 'experiments_' + str(x) + '_' + str(y) + '/test/'
    Output_Data = constants.OUTPUT_DIR + 'predictions/' + current_data
    thresholds = np.arange(precision) / precision
    models = ['LGBM', 'LSTM', 'CoxPHM']
    pred_sepsispatient_sublist = []
    pred_sepsispatient_sublist_list = []
    true_septic_perpatient_list = []

    for definition in definitions:
        print(definition)
        pred_sepsispatient_sublist = []
        path_df = constants.DATA_DIR + '/raw/further_split/val_' + str(x) + '_' + str(y) + '.csv'
        df_sepsis1 = pd.read_pickle(Root_Data + definition[1:] + '_dataframe.pkl')
        current_label = np.load(Root_Data + 'label' + definition[1:] + '_' + str(T) + '.npy')
        for model in models:
            print(model)

            prob_preds = np.load(
                Output_Data + model + '/' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:] + '.npy')
            print(df_sepsis1.shape, current_label.shape, prob_preds.shape)
            pred_labels = (prob_preds > 0).astype('int')
            patient_true_label, _, _, _, _, _, _ = patient_level_pred(df_sepsis1, current_label, pred_labels, T,
                                                                      sample_ids=None, cohort='full')
            true_septic_perpatient_list.append(patient_true_label.values)
            CMs, patient_pred_label_list, _ = suboptimal_choice_patient(df_sepsis1, current_label,
                                                                        prob_preds, a1=T,
                                                                        thresholds=thresholds,
                                                                        sample_ids=None)
            patient_pred_label_at_levels, precision, tpr, tnr, _ = output_metric_level(CMs,
                                                                                       patient_pred_label_list,
                                                                                       levels=[metric_thresh],
                                                                                       test_metric=test_metric)
            print(precision, tpr, tnr)
            pred_sepsispatient_sublist.append(patient_pred_label_at_levels)
        pred_sepsispatient_sublist_list.append(pred_sepsispatient_sublist)
    # plot
    Definitions = ['H1', 'H2', 'H3']
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs = axs.ravel()
    for i in range(3):
        a = np.where(pred_sepsispatient_sublist_list[i][0][0] == 1)[0]
        b = np.where(pred_sepsispatient_sublist_list[i][1][0] == 1)[0]
        c = np.where((true_septic_perpatient_list[3 * i] == 1))[0]

        venn_values = venn_3counts(a, b, c)

        venn3(subsets=venn_values, set_labels=('LGBM', 'LSTM', 'True'), alpha=0.5, ax=axs[i])

        axs[i].set_title(Definitions[i])
        # plt.show()
    plt.tight_layout()
    plt.savefig(save_dir + 'Venn_diagram_compare_models' + '.png')


def sepsis_onset_time_plots(x, y, T, test_metric, metric_thresh, precision, save_dir, strict_exclusion=False):
    definitions = ['t_sofa', 't_suspicion', 't_sepsis_min']
    models = ['LGBM', 'LSTM', 'CoxPHM']
    current_data = 'blood_only_data/'
    Root_Data = constants.DATA_processed + current_data + 'experiments_' + str(x) + '_' + str(y) + '/test/'
    Output_Data = constants.OUTPUT_DIR + 'predictions/' + current_data
    thresholds = np.arange(precision) / precision

    if strict_exclusion:
        df_sepsis1 = pd.read_pickle(Root_Data + '_sepsis_min' + '_dataframe.pkl')
        sample_ids = df_sepsis1['icustay_id'].unique()
    else:
        sample_ids = None

    true_septic_time_list = []
    pred_septic_time_sublists = []
    ids_list = []
    patient_true_label_list = []
    patient_icustay_list = []
    septic_los_list = []
    for definition in definitions:
        df_sepsis1 = pd.read_pickle(Root_Data + definition[1:] + '_dataframe.pkl')
        septic_los_list.append(
            df_sepsis1.loc[df_sepsis1['sepsis_hour'].notnull()].groupby(
                'icustay_id').rolling_los_icu.max().astype(
                'int'))
        current_label = np.load(Root_Data + 'label' + definition[1:] + '_' + str(T) + '.npy')
        for model in models:
            prob_preds = np.load(
                Output_Data + model + '/' + str(x) + '_' + str(y) + '_' + str(T) + definition[
                                                                                   1:] + '_test' + '.npy')

            CMs, patient_pred_label_list, pred_septic_time_list = suboptimal_choice_patient(df_sepsis1,
                                                                                            current_label,
                                                                                            prob_preds, a1=6,
                                                                                            thresholds=thresholds,
                                                                                            sample_ids=sample_ids)
            _, _, _, _, idx = output_metric_level(CMs, patient_pred_label_list, levels=[metric_thresh],
                                                  test_metric=test_metric)
            threshold = thresholds[idx]
            # print(df_sepsis1.shape,current_labels.shape,pred_labels.shape,prob_preds.shape)
            pred_labels = (prob_preds > threshold).astype('int')
            patient_true_label, _, _, pred_septic_time, true_septic_time, ids, patient_icustay = patient_level_pred(
                df_sepsis1, current_label,
                pred_labels, T, sample_ids=sample_ids,
                cohort='full')
            patient_icustay_list.append(patient_icustay)
            ids_list.append(ids)
            patient_true_label_list.append(patient_true_label)
            true_septic_time_list.append(true_septic_time)
            pred_septic_time_sublists.append(pred_septic_time)
    true_id_list = [true_septic_time_list[i].index for i in range(len(true_septic_time_list))]
    identified_pred_sepsis_time = [
        pred_septic_time_sublists[i].loc[pred_septic_time_sublists[i].index.isin(true_id_list[i])]
        for i in range(len(pred_septic_time_sublists))]
    time_difference_dist_definitions1(true_septic_time_list, identified_pred_sepsis_time,
                                      time_grid=[0, 6, 12, 18, 24, 30, 36, 42, 48], save_dir=save_dir)

    # median_time_difference(true_septic_time_list, identified_pred_sepsis_time, time_grid=np.arange(6, 16),
    #             save_dir=save_dir)
    # median_time_difference_interval(true_septic_time_list, identified_pred_sepsis_time,
    # time_grid=[(4, 6), (6, 8), (8, 10), (10, 12),
    #         (12, 14), (14, 300)], save_dir=save_dir)
    onset_time_stacked_bar(patient_true_label_list, true_septic_time_list, identified_pred_sepsis_time,
    pred_septic_time_sublists,
    patient_icustay_list, septic_los_list, time_grid=np.arange(41),
    save_dir=save_dir, definition='H1')
    onset_time_stacked_bar(patient_true_label_list, true_septic_time_list, identified_pred_sepsis_time,
    pred_septic_time_sublists,
    patient_icustay_list, septic_los_list, time_grid=np.arange(41),
    save_dir=save_dir, definition='H2')
    onset_time_stacked_bar(patient_true_label_list, true_septic_time_list, identified_pred_sepsis_time,
    pred_septic_time_sublists,
    patient_icustay_list, septic_los_list, time_grid=np.arange(41),
    save_dir=save_dir, definition='H3')
    # proprotion_HBO_line_plot(patient_true_label_list, true_septic_time_list, identified_pred_sepsis_time,
    # patient_icustay_list,
    # time_grid=np.arange(61), save_dir=save_dir)


def time_difference_dist_model(true_septic_time_list, identified_pred_sepsis_time,
                               time_grid=[0, 6, 12, 18, 24, 30, 36, 42, 48], save_dir=None):
    models = []
    proportion = []
    time_diff = []
    model_grid = ['LGBM', 'LSTM', 'CoxPHM']

    for i in range(3):
        identified_true_sepsis_time = true_septic_time_list[i].loc[
            true_septic_time_list[i].index.isin(identified_pred_sepsis_time[i].index)]
        identified_time_difference = identified_true_sepsis_time.values - identified_pred_sepsis_time[i].values
        for time in time_grid:
            time_diff.append(r'$\geq' + str(time))
            models.append(model_grid[i])

            proportion.append(
                np.where(identified_time_difference > time)[0].shape[0] / len(true_septic_time_list[i]))
    data_dict = {'HBO': time_diff, 'model': models, 'proportion': proportion}
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    sns.barplot(x="HBO", y="proportion", hue="model", data=pd.DataFrame.from_dict(data_dict))
    plt.savefig(save_dir + 'Time_diff_dist_models' + '.png')


def time_difference_dist_definitions(true_septic_time_list, identified_pred_sepsis_time,
                                     time_grid=[0, 6, 12, 18, 24, 30, 36, 42, 48], save_dir=None):
    defs = []
    proportion = []
    time_diff = []
    def_grid = ['H1', 'H2', 'H3']

    for i in range(3):
        identified_true_sepsis_time = true_septic_time_list[3 * i].loc[
            true_septic_time_list[3 * i].index.isin(identified_pred_sepsis_time[3 * i].index)]
        identified_time_difference = identified_true_sepsis_time.values - identified_pred_sepsis_time[
            3 * i].values
        for time in time_grid:
            time_diff.append(r'$\geq$' + str(time))
            defs.append(def_grid[i])

            proportion.append(
                np.where(identified_time_difference > time)[0].shape[0] / len(true_septic_time_list[3 * i]))
        # print(len(true_septic_time_list[3 * i]))
    data_dict = {'HBO': time_diff, 'def': defs, 'proportion': proportion}
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 17})
    sns.barplot(x="HBO", y="proportion", hue="def", data=pd.DataFrame.from_dict(data_dict))
    plt.savefig(save_dir + 'Time_diff_dist_definitions' + '.jpeg', dpi=350)


def time_difference_dist_definitions1(true_septic_time_list, identified_pred_sepsis_time,
                                      time_grid=[0, 6, 12, 18, 24, 30, 36, 42, 48], save_dir=None):
    defs = []
    proportion = []
    time_diff = []
    def_grid = ['H1', 'H2', 'H3']

    for i in range(3):
        identified_true_sepsis_time = true_septic_time_list[3 * i].loc[
            true_septic_time_list[3 * i].index.isin(identified_pred_sepsis_time[3 * i].index)]
        identified_time_difference = identified_true_sepsis_time.values - identified_pred_sepsis_time[
            3 * i].values
        for time in time_grid:
            time_diff.append(r'$\geq$' + str(time))
            defs.append(def_grid[i])

            proportion.append(
                np.where(identified_time_difference > time)[0].shape[0] / len(true_septic_time_list[3 * i]))
        # print(len(true_septic_time_list[3 * i]))
    defs = []
    proportion1 = []

    time_diff = []
    def_grid = ['H1', 'H2', 'H3']

    for i in range(3):
        for time in time_grid:
            time_diff.append(r'$\geq$' + str(time))
            defs.append(def_grid[i])
            patient_proportion = (np.where(true_septic_time_list[3 * i] > time)[0].shape[0]) / len(
                true_septic_time_list[3 * i])
            proportion1.append(patient_proportion)

    data_dict = {'HBO': time_diff, 'def': defs, 'proportion': proportion}
    data_dict1 = {'HBO': time_diff, 'def': defs,
                  'proportion': np.array(proportion1)}
    color_pal = sns.color_palette("colorblind", 3).as_hex()
    colors = ','.join(color_pal)
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 17})
    sns.barplot(x="HBO", y="proportion", hue="def", data=pd.DataFrame.from_dict(data_dict))
    sns.lineplot(x="HBO", y="proportion",
                 hue="def",
                 data=pd.DataFrame.from_dict(data_dict1), style='def', sort=False,
                 markers=['o', 'o', 'o'],
                 dashes=[(5, 5), (5, 5), (5, 5)])

    plt.savefig(save_dir + 'Time_diff_dist_definitions' + '.jpeg', dpi=350)


def proprotion_HBO_line_plot(patient_true_label_list, true_septic_time_list, identified_pred_sepsis_time,
                             patient_icustay_list,
                             time_grid=np.arange(0, 61), save_dir=None):
    defs = []
    proportion = []

    time_diff = []
    defs1 = []
    def_grid = ['H1', 'H2', 'H3']
    patience_def = ['flagged as septic', 'true septic']

    for i in range(3):
        identified_true_sepsis_time = true_septic_time_list[3 * i].loc[
            true_septic_time_list[3 * i].index.isin(identified_pred_sepsis_time[3 * i].index)]
        identified_time_difference = identified_true_sepsis_time.values - identified_pred_sepsis_time[
            3 * i].values
        fn_ids = [x for x in true_septic_time_list[3 * i].index if
                  x not in identified_pred_sepsis_time[3 * i].index]
        fn_icustay = patient_icustay_list[3 * i].loc[fn_ids]
        negative_ids = [x for x in patient_true_label_list[3 * i].index if
                        x not in true_septic_time_list[3 * i].index]
        for time in time_grid:
            for item in patience_def:
                time_diff.append(str(time))
                defs.append(def_grid[i])
                defs1.append(item)
                patient_proportion = (np.where(true_septic_time_list[3 * i] > time)[0].shape[0]) / len(
                    true_septic_time_list[3 * i])
                tp_proportion = np.where(identified_time_difference == time)[0].shape[0] / len(
                    true_septic_time_list[3 * i])
                proportion.append(tp_proportion) if item == 'flagged as septic' else proportion.append(
                    patient_proportion)

        # print(time_diff)
        # print(len(true_septic_time_list[3 * i]))

    data_dict1 = {'Hours before sepsis onset': time_diff, 'def': defs,
                  'proportion (septic patients in ICU / septic patients)': np.array(proportion),
                  'cohort': defs1}
    color_pal = sns.color_palette("colorblind", 6).as_hex()
    colors = ','.join(color_pal)
    # print(pd.DataFrame.from_dict(data_dict1))
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 17, 'legend.handlelength': 1.2})
    sns.lineplot(x="Hours before sepsis onset", y="proportion (septic patients in ICU / septic patients)",
                 hue="def",
                 data=pd.DataFrame.from_dict(data_dict1), style='cohort', sort=False, color=colors)
    # plt.title('proportion of predicted septic and septic+ non_septic conditional on septic patients')
    locs, labels = plt.xticks()
    # ax.set_ylabel('proportion (patients in ICU/ total patients)')
    plt.xticks(np.arange(0, time_grid[-1] + 1, step=5))
    plt.savefig(save_dir + 'proprotion_HBO_line_plot' + '.jpeg', dpi=350)
    return pd.DataFrame.from_dict(data_dict1)


def onset_time_dist_definitions(true_septic_time_list, identified_pred_sepsis_time,
                                time_grid=np.arange(6, 16), save_dir=None):
    defs = []
    proportion = []
    icustay = []
    def_grid = ['H1', 'H2', 'H3']

    for i in range(3):
        identified_true_sepsis_time = true_septic_time_list[3 * i].loc[
            true_septic_time_list[3 * i].index.isin(identified_pred_sepsis_time[3 * i].index)]
        for time in time_grid:
            icustay.append('>=' + str(time))
            defs.append(def_grid[i])

            proportion.append(
                np.where(identified_true_sepsis_time >= time)[0].shape[0] / len(true_septic_time_list[3 * i]))
        # print(len(true_septic_time_list[3 * i]))
    data_dict = {'icustay': icustay, 'def': defs, 'proportion': proportion}
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    sns.barplot(x='icustay', y="proportion", hue="def", data=pd.DataFrame.from_dict(data_dict))
    plt.savefig(save_dir + 'onset_time_dist_definitions' + '.png')


def median_time_difference_interval(true_septic_time_list, identified_pred_sepsis_time,
                                    time_grid=[(4, 6), (6, 8), (8, 10), (10, 12),
                                               (12, 14), (14, 1000)], save_dir=None):
    model_def = []
    median_list = []
    icustay = []
    def_grid = ['H1', 'H2', 'H3']
    model_grid = ['LGBM', 'LSTM', 'CoxPHM']
    for i in range(len(def_grid)):
        for j in range(len(model_grid)):
            identified_true_sepsis_time = true_septic_time_list[3 * i + j].loc[
                true_septic_time_list[3 * i + j].index.isin(identified_pred_sepsis_time[3 * i + j].index)]
            for k, time in enumerate(time_grid):
                ids = np.where((identified_true_sepsis_time.values >= time[0]) & (
                        identified_true_sepsis_time.values <= time[1]))[0]
                median_list.append(np.median(
                    identified_true_sepsis_time.values[ids] - identified_pred_sepsis_time[3 * i + j].values[
                        ids]))
                model_def.append(def_grid[i] + '&' + model_grid[j])
                # icustay.append(r'['+str(time[0])+','+str(time[1])+r']') if k != len(time_grid)-1 \
                # else icustay.append(r'['+str(time[0])+','+str(max(true_septic_time_list[3 * i + j]))+r']')
                icustay.append(r'[' + str(time[0]) + ',' + str(time[1]) + r']')
    data_dict = {'icustay': icustay, 'model_definitions': model_def, 'median of HBO': median_list}
    df = pd.DataFrame.from_dict(data_dict)
    df['definition'] = [item[:2] for item in df['model_definitions'].values]
    df['model'] = [item[3:] for item in df['model_definitions'].values]
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 17})
    sns.lineplot(data=df, x='icustay', y='median of HBO', hue='definition', style='model', markers=True,
                 sort=False)
    plt.legend(loc='upper left', prop={'size': 17})
    plt.savefig(save_dir + 'median of HBO different icustays' + '.jpeg', dpi=350)


def true_onset_time_dist(true_septic_time_list, time_grid=np.arange(0, 50)):
    def_grid = ['H1', 'H2', 'H3']
    icustay = []
    proportion = []
    defs = []
    for i in range(3):
        for time in time_grid:
            icustay.append(time)
            proportion.append(np.where(true_septic_time_list[3 * i] >= time)[0].shape[0])
            defs.append(def_grid[i])
    data_dict = {'def': defs, 'number of septic patient': proportion, 'icustay': icustay}
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    sns.lineplot(data=pd.DataFrame.from_dict(data_dict), x='icustay', y='number of septic patient', hue='def')


def median_time_difference(true_septic_time_list, identified_pred_sepsis_time, time_grid=np.arange(6, 16),
                           save_dir=None):
    model_def = []
    median_list = []
    icustay = []
    def_grid = ['H1', 'H2', 'H3']
    model_grid = ['LGBM', 'LSTM', 'CoxPHM']
    for i in range(len(def_grid)):
        for j in range(len(model_grid)):
            identified_true_sepsis_time = true_septic_time_list[3 * i + j].loc[
                true_septic_time_list[3 * i + j].index.isin(identified_pred_sepsis_time[3 * i + j].index)]
            for time in time_grid:
                ids = np.where(identified_true_sepsis_time.values >= time)[0]
                median_list.append(np.median(
                    identified_true_sepsis_time.values[ids] - identified_pred_sepsis_time[3 * i + j].values[
                        ids]))
                model_def.append(def_grid[i] + '&' + model_grid[j])
                icustay.append(r'$\geq$' + str(time))
    data_dict = {'icustay': icustay, 'model_definitions': model_def, 'median of HBO': median_list}
    df = pd.DataFrame.from_dict(data_dict)
    df['definition'] = [item[:2] for item in df['model_definitions'].values]
    df['model'] = [item[3:] for item in df['model_definitions'].values]
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 17})
    sns.lineplot(data=df, x='icustay', y='median of HBO', hue='definition', style='model', markers=True,
                 sort=False)
    plt.legend(loc='lower right', prop={'size': 17})
    plt.savefig(save_dir + 'median of HBO' + '.jpeg', dpi=350)


def median_flagtime_on_true(true_septic_time_list, identified_pred_sepsis_time, time_grid=np.arange(6, 16),
                            save_dir=None):
    model_def = []
    median_list = []
    icustay = []
    def_grid = ['H1', 'H2', 'H3']
    model_grid = ['LGBM', 'LSTM', 'CoxPHM']
    for i in range(len(def_grid)):
        for j in range(len(model_grid)):
            identified_true_sepsis_time = true_septic_time_list[3 * i + j].loc[
                true_septic_time_list[3 * i + j].index.isin(identified_pred_sepsis_time[3 * i + j].index)]
            for time in time_grid:
                identified_pred_sepsis_time[np.where(identified_true_sepsis_time == time)]


def onset_time_stacked_bar(patient_true_label_list, true_septic_time_list, identified_pred_sepsis_time,
                           pred_septic_time_list,
                           patient_icustay_list, septic_los_list, time_grid=np.arange(41),
                           save_dir=None, definition='H1'):
    defs = ['H1', 'H2', 'H3']
    i = np.where(np.array(defs) == definition)[0][0]
    tp_proportion = []
    fn_proportion = []
    tn_proportion = []
    fp_proportion = []
    tp_discharge_proportion = []
    icustay = []
    # tp_sepsis_time = true_septic_time_list[i].loc[
    # true_septic_time_list[i].index.isin(identified_pred_sepsis_time[i].index)]
    tp_sepsis_time = identified_pred_sepsis_time[i].values
    fn_ids = [x for x in true_septic_time_list[i].index if x not in identified_pred_sepsis_time[i].index]
    fn_icustay = patient_icustay_list[i].loc[fn_ids]
    negative_ids = [x for x in patient_true_label_list[i].index if x not in true_septic_time_list[i].index]
    tp_icustay = true_septic_time_list[i].loc[
        true_septic_time_list[i].index.isin(identified_pred_sepsis_time[i].index)]
    tp_icustay_discharge = septic_los_list[i].loc[
        septic_los_list[i].index.isin(identified_pred_sepsis_time[i].index)]

    fp_sepsis_time = pred_septic_time_list[i].loc[pred_septic_time_list[i].index.isin(negative_ids)]
    fp_icustay = patient_icustay_list[i].loc[patient_icustay_list[i].index.isin(negative_ids)]
    tn_ids = [x for x in negative_ids if x not in fp_sepsis_time.index]
    tn_icustay = patient_icustay_list[i].loc[tn_ids]
    time_diff = time_grid[1] - time_grid[0]
    for time in time_grid:
        tp_proportion.append(
            (np.where(tp_sepsis_time <= time)[0].shape[0] - np.where(tp_icustay < time)[0].shape[0]) / len(
                patient_true_label_list[i]))
        fp_proportion.append(
            (np.where(fp_sepsis_time <= time)[0].shape[0] - np.where(fp_icustay < time)[0].shape[0]) / len(
                patient_true_label_list[i]))
        fn_proportion.append(
            (np.where(tp_sepsis_time > time)[0].shape[0] + len(fn_ids) - np.where(fn_icustay < time)[0].shape[
                0]) / len(
                patient_true_label_list[i]))
        tn_proportion.append(
            (np.where(fp_sepsis_time > time)[0].shape[0] + len(tn_ids) - np.where(tn_icustay < time)[0].shape[
                0]) / len(
                patient_true_label_list[i]))
        tp_discharge_proportion.append(
            (np.where(tp_sepsis_time <= time)[0].shape[0] - np.where(tp_icustay_discharge < time)[0].shape[
                0]) / len(
                patient_true_label_list[i]))
        icustay.append(str(time))
    icustay[0] = str(0)
    data_dict = {'Hours since icu admission': icustay,
                 'flagged,ultimately septic': tp_proportion,
                 'flagged,septic until discharge': [tp_discharge_proportion[i] - tp_proportion[i] for i in
                                                    range(len(tp_proportion))],
                 'flagged,ultimately non-septic': fp_proportion,
                 'unflagged,ultimately non-septic': tn_proportion,
                 'unflagged,ultimately septic': fn_proportion}
    df = pd.DataFrame.from_dict(data_dict)
    df1 = df.T
    new_header = df1.iloc[0]
    df1 = df1[1:]  # take the data less the header row
    df1.columns = new_header
    plt.figure(figsize=(30, 14))
    # plt.rcParams.update({'font.size': 30})
    params = {'legend.fontsize': 17, 'legend.handlelength': 0.5, 'axes.labelsize': 17}
    plt.rcParams.update(params)
    color_pal = sns.color_palette("colorblind", 6).as_hex()
    colors = ','.join(color_pal)
    ax = df1.T.plot(kind='bar', stacked=True, figsize=(12, 9), fontsize=14, rot=0, legend=True, color=color_pal)
    ax.set_title(definition)
    ax.set_ylabel('proportion (patients in ICU/ total patients)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))
    for i, t in enumerate(ax.get_xticklabels()):
        if (i % 3) != 0:
            t.set_visible(False)
    # patches, labels = ax.get_legend_handles_labels()
    # ax.legend(patches, labels, loc='best')
    plt.savefig(save_dir + 'outcome_stacked_bar_plot_' + definition + '.jpeg', dpi=350)
    return df1


def proprotion_HBO_line_plot(patient_true_label_list, true_septic_time_list, identified_pred_sepsis_time,
                             patient_icustay_list,
                             time_grid=np.arange(0, 61), save_dir=None):
    defs = []
    proportion = []

    time_diff = []
    defs1 = []
    def_grid = ['H1', 'H2', 'H3']
    patience_def = ['flagged as septic', 'true septic']

    for i in range(3):
        identified_true_sepsis_time = true_septic_time_list[3 * i].loc[
            true_septic_time_list[3 * i].index.isin(identified_pred_sepsis_time[3 * i].index)]
        identified_time_difference = identified_true_sepsis_time.values - identified_pred_sepsis_time[
            3 * i].values
        fn_ids = [x for x in true_septic_time_list[3 * i].index if
                  x not in identified_pred_sepsis_time[3 * i].index]
        fn_icustay = patient_icustay_list[3 * i].loc[fn_ids]
        negative_ids = [x for x in patient_true_label_list[3 * i].index if
                        x not in true_septic_time_list[3 * i].index]
        for time in time_grid:
            for item in patience_def:
                time_diff.append(str(time))
                defs.append(def_grid[i])
                defs1.append(item)
                patient_proportion = (np.where(true_septic_time_list[3 * i] > time)[0].shape[0]) / len(
                    true_septic_time_list[3 * i])
                tp_proportion = np.where(identified_time_difference == time)[0].shape[0] / len(
                    true_septic_time_list[3 * i])
                proportion.append(tp_proportion) if item == 'flagged as septic' else proportion.append(
                    patient_proportion)

        # print(time_diff)
        # print(len(true_septic_time_list[3 * i]))

    data_dict1 = {'Hours before sepsis onset': time_diff, 'def': defs,
                  'proportion (septic patients in ICU / septic patients)': np.array(proportion),
                  'cohort': defs1}
    color_pal = sns.color_palette("colorblind", 6).as_hex()
    colors = ','.join(color_pal)
    # print(pd.DataFrame.from_dict(data_dict1))
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 17, 'legend.handlelength': 1.2})
    sns.lineplot(x="Hours before sepsis onset", y="proportion (septic patients in ICU / septic patients)",
                 hue="def",
                 data=pd.DataFrame.from_dict(data_dict1), style='cohort', sort=False, color=colors)
    # plt.title('proportion of predicted septic and septic+ non_septic conditional on septic patients')
    locs, labels = plt.xticks()
    # ax.set_ylabel('proportion (patients in ICU/ total patients)')
    plt.xticks(np.arange(0, time_grid[-1] + 1, step=5))
    plt.savefig(save_dir + 'proprotion_HBO_line_plot' + '.jpeg', dpi=350)
    return pd.DataFrame.from_dict(data_dict1)


def auc_plots(definition_list, model_list, save_dir=constants.OUTPUT_DIR + 'plots/', T=6, train_test='test'):
    precision, n, a1 = 100, 100, 6
    current_data = 'blood_only_data/'

    Output_Data = constants.OUTPUT_DIR + 'predictions/' + current_data
    # mimic3_myfunc.create_folder(Data_save)
    for definition in definition_list:
        for model in model_list:
            labels_list = []
            probs_list = []
            tprs_list = []
            fprs_list = []
            for x, y in constants.xy_pairs:
                print(definition, x, y, model)
                Data_Dir = constants.DATA_processed + current_data + 'experiments_' + str(x) + '_' + str(
                    y) + '/' + train_test + '/'

                labels_now = np.load(Data_Dir + 'label' + definition[1:] + '_6.npy')

                if model == 'LSTM' or model == 'CoxPHM':
                    probs_now = np.load(
                        Output_Data + model + '/' + str(x) + '_' + str(y) + '_' + str(T) + definition[
                                                                                           1:] + '_' + train_test + '.npy')
                else:
                    probs_now = np.load(
                        Output_Data + model + '/' + 'prob_preds' + '_' + str(x) + '_' + str(y) + '_' + str(
                            T) + '_' + definition[1:] + '.npy')

                icu_lengths_now = np.load(Data_Dir + 'icustay_lengths' + definition[1:] + '.npy')

                icustay_fullindices_now = mimic3_myfunc.patient_idx(icu_lengths_now)

                tpr, fpr = mimic3_myfunc_patientlevel.patient_level_auc(labels_now, probs_now,
                                                                        icustay_fullindices_now,
                                                                        precision, n=n, a1=a1)

                labels_list.append(labels_now)
                probs_list.append(probs_now)
                tprs_list.append(tpr)
                fprs_list.append(fpr)

            names = ['48,24', '24,12', '12,6', '6,3']

            auc_plot(labels_list, probs_list, names=names, \
                     save_name=save_dir + 'auc_plot_instance_level_' + model + definition[
                                                                               1:] + '_' + train_test)
            auc_plot_patient_level(fprs_list, tprs_list, names=names, \
                                   save_name=save_dir + 'auc_plot_patient_level_' + model + definition[
                                                                                            1:] + '_' + train_test)
