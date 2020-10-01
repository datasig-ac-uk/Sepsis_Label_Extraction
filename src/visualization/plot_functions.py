import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

from matplotlib.patches import Rectangle

from src.features.sepsis_mimic3_myfunction import patient_level_performance_new


def barplot_two(data1, data2, \
                x_label, y_label, \
                title, labels, \
                y_tick_percentage=False, \
                label1='$\mathcal{D}_1$', label2='$\mathcal{D}_2$', \
                width=0.4, \
                figsize=(15, 8), \
                text_appearance='{0:.1%}', \
                fontsize=16, \
                savetitle=None):
    """

        plot Barplot from two bunches of data in the same plot.


    Parameters
    ----------
    data1: the first set of data
    data2: the second set of data

    label1: label for data1
    label2: label for data2

    x_label: xaxis label
    y_label: yaxis label

    width: the width of the bars
    figsize: figsize of the fig
    text_appearance: the appearance of text
    fontsize: fontsize for labels



    """

    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots(figsize=figsize)

    rects1 = ax.bar(x - width, data1, width, label=label1)
    rects2 = ax.bar(x, data2, width, label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=fontsize - 3)
    ax.tick_params(axis="y", labelsize=fontsize - 3)
    ax.set_xticklabels(labels)

    ax.legend(fontsize=fontsize)
    if y_tick_percentage:
        #         fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
        #         yticks = mtick.FormatStrFormatter(fmt)
        #         ax.yaxis.set_major_formatter(yticks)
        yticks = mtick.FuncFormatter("{:.2%}".format)
        ax.yaxis.set_major_formatter(yticks)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(text_appearance.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    if savetitle != None:
        plt.savefig("./" + savetitle + '.png')
    else:
        plt.show()


def barplot_three(data1, data2, data3, \
                  data_labels, \
                  x_label, y_label, \
                  title, labels, \
                  y_tick_percentage=False, \
                  width=0.4, \
                  figsize=(10, 6), \
                  text_appearance='{0:.1%}', \
                  fontsize=15, \
                  Directory_save='./Sep_plots/', \
                  savetitle=None, \
                  extra_text_legend=True):
    """

        plot Barplot from two bunches of data in the same plot.


    Parameters
    ----------
    data1: the first set of data
    data2: the second set of data

    label1: label for data1
    label2: label for data2

    x_label: xaxis label
    y_label: yaxis label

    width: the width of the bars
    figsize: figsize of the fig
    text_appearance: the appearance of text
    fontsize: fontsize for labels



    """

    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots(figsize=figsize)

    rects1 = ax.bar(x - width * 2 / 3, data1, width / 2)
    rects2 = ax.bar(x - width / 6, data2, width / 2)
    rects3 = ax.bar(x + width / 3, data3, width / 2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=fontsize - 2)
    ax.tick_params(axis="y", labelsize=fontsize - 2)
    ax.set_xticklabels(labels, fontsize=fontsize)

    if extra_text_legend:
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        ax.legend([extra, rects1, rects2, rects3], \
                  (data_labels[0], data_labels[1], data_labels[2], data_labels[3]), fontsize=fontsize - 2)
    else:
        ax.legend([rects1, rects2, rects3], (data_labels[0], data_labels[1], data_labels[2]), fontsize=fontsize - 2)
    if y_tick_percentage:
        #         fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
        #         yticks = mtick.FormatStrFormatter(fmt)
        #         ax.yaxis.set_major_formatter(yticks)
        yticks = mtick.FuncFormatter("{:.2%}".format)
        ax.yaxis.set_major_formatter(yticks)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(text_appearance.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    if savetitle != None:
        plt.savefig(Directory_save + savetitle + '.png', dpi=300)
    else:
        plt.show()


def septic_proportions_compare(icu_number_old_df, icu_number_new_df, \
                               savetitle, \
                               definition='t_sofa', \
                               x_label='x,y', \
                               y_label='Proportion of septic ICUstay', \
                               labels=['48,24', '24,12', '12,6', '6,3']):
    sorterIndex = dict(zip(labels, range(len(labels))))

    icu_number_old['x,y'] = icu_number_old['x,y'].astype("category")
    icu_number_old['x,y'].cat.set_categories(sorter, inplace=True)

    icu_number_new['x,y'] = icu_number_new['x,y'].astype("category")
    icu_number_new['x,y'].cat.set_categories(sorter, inplace=True)

    data1 = np.array(icu_number_old[icu_number_old.definition == definition].sort_values(by=['x,y'])['septic ratio'])
    data2 = np.array(icu_number_new[icu_number_new.definition == definition].sort_values(by=['x,y'])['septic ratio'])

    barplot_two(data1, data2, \
                x_label, y_label, \
                title, labels, \
                y_tick_percentage=True, \
                width=0.4 / 1.5, \
                figsize=(10, 6), \
                text_appearance='{0:.2%}', \
                fontsize=16, \
                savetitle=savetitle)


def stacked_barplot3lists_compare(data1, data2, data3, \
                                  label1, label2, label3, \
                                  xlabel, ylabel, \
                                  labels, \
                                  width=0.4, \
                                  fontsize=14, \
                                  figsize=(15, 8), \
                                  savetitle=None):
    sns.set(color_codes=True)
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width * 2 / 3, data1[0], width / 2, label=label1[0], hatch="X", color=sns.color_palette()[0])
    rects1_ = ax.bar(x - width * 2 / 3, data1[1], width / 2, bottom=data1[0], label=label1[1],
                     color=sns.color_palette()[0])
    rects2 = ax.bar(x - width / 6, data2[0], width / 2, label=label2[0], hatch="X", color=sns.color_palette()[1])
    rects2_ = ax.bar(x - width / 6, data2[1], width / 2, bottom=data2[0], label=label2[1], color=sns.color_palette()[1])

    rects3 = ax.bar(x + width / 3, data3[0], width / 2, label=label3[0], hatch="X", color=sns.color_palette()[2])
    rects3_ = ax.bar(x + width / 3, data3[1], width / 2, bottom=data3[0], label=label3[1], color=sns.color_palette()[2])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.set_xticks(x)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_xticklabels(labels, fontsize=fontsize)

    ax.tick_params(axis="x", labelsize=fontsize - 3)
    ax.tick_params(axis="y", labelsize=fontsize - 3)
    ax.legend(fontsize=fontsize, bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)

    fig.tight_layout()

    if savetitle == None:
        plt.show()
    else:
        plt.savefig(savetitle + ".eps")


# colors=['darkorange','darkcyan','deeppink','lightnavy']

colors = sns.color_palette()
linestyles = [':', '-.', '-', '--']


def plot_confusions(cm, target_names, xlabel, ylabel, \
                    figsize=(5, 3), fontsize=16, \
                    Directory_save='./Sep_plots/', savetitle=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    df_cm1 = pd.DataFrame(cm, target_names,
                          target_names)
    sns.set(font_scale=1.0)  # for label size
    sns.heatmap(df_cm1, cmap="Blues", cbar=False, annot=True, annot_kws={"size": fontsize}, fmt='g', ax=ax)  # font size

    if savetitle is not None:
        plt.savefig(Directory_save + savetitle + '.png', dpi=300)
    else:

        plt.show()


def plot_confusions_normalised(cm1, target_names, xlabel, ylabel, \
                               figsize=(5, 3), fontsize=16, \
                               Directory_save='./Sep_plots/', savetitle=None):
    cm = cm1 / np.repeat(cm1.sum(axis=1), 2).reshape(2, 2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    df_cm1 = pd.DataFrame(cm, target_names,
                          target_names)
    sns.set(font_scale=1.0)  # for label size
    sns.heatmap(df_cm1, cmap="Blues", cbar=False, annot=True, fmt='.2%', annot_kws={"size": fontsize},
                ax=ax)  # font size

    if savetitle is not None:
        plt.savefig(Directory_save + savetitle + '.png', dpi=300)
    else:

        plt.show()


def auc_plot(trues_list, probs_list, names, \
             fontsize=14, \
             colors=colors, \
             linestyles=linestyles, \
             save_name=None):
    num = len(trues_list)
    plt.figure()
    lw = 2
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
    #     plt.title('Receiver operating characteristic of lgbm')
    plt.legend(loc="lower right", fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    if save_name is not None:
        plt.savefig(save_name + '.png')
    else:

        plt.show()


def rect_line_at_turn(path, turn_value=1, replace_value=0):
    num = len(path)

    time_seq = np.arange(num)

    repeated_idx = np.where(path == turn_value)[0][0]

    new_path = np.insert(path, repeated_idx, 0)
    new_time_seq = np.insert(time_seq, repeated_idx, time_seq[repeated_idx])

    true_labels = np.zeros(len(new_path))
    true_labels[-1] = 1

    true_times = np.arange(len(new_path))
    true_times[-1] = true_times[-2]

    return new_time_seq, new_path, true_times, true_labels


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


def sepsis_time_error_plot(thresholds, mean_time_list, optimal_thresh_list, names=['H1', 'H2', 'H3'], \
                           fontsize=14, colors=sns.color_palette(), save_name=None):
    num = len(mean_time_list)
    plt.figure()
    lw = 2
    for i in range(num):
        plt.plot(thresholds, mean_time_list[i], color=colors[i], linestyle=linestyles[i], \
                 lw=lw, label=names[i])
        plt.vlines(optimal_thresh_list[i], ymin=-5, ymax=55, color=colors[i])
        plt.xlabel('thresholds', fontsize=fontsize)
        plt.ylabel('mean septic time error ', fontsize=fontsize)
        plt.legend(loc="upper left", fontsize=fontsize - 3)
        plt.xticks(fontsize=fontsize - 3)
        plt.yticks(fontsize=fontsize - 3)
        if save_name is not None:
            plt.savefig(save_name + '.png')
        else:

            plt.show()


def patient_level_auc(labels_true, train_preds, test_full_indices, precision):
    tpr, fpr = [], []
    thresholds = np.arange(precision) / precision
    for thresh in thresholds:
        preds = (train_preds >= thresh).astype('int')

        _, _, _, CM = patient_level_performance_new(preds, labels_true, test_full_indices, k=5)
        FP = CM[0, 1]
        FN = CM[1, 0]
        TP = CM[1, 1]
        TN = CM[0, 0]
        tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))
    return tpr, fpr, thresholds


def auc_plot_patient_level(trues_list, probs_list, test_full_indices_list, names, \
                           fontsize=14, \
                           colors=colors, \
                           linestyles=linestyles, \
                           save_name=None):
    num = len(trues_list)
    plt.figure()
    lw = 2
    for i in range(num):
        tpr, fpr, _ = patient_level_auc(trues_list[i], probs_list[i], test_full_indices_list[i], precision=100)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i], linestyle=linestyles[i], \
                 lw=lw, label='ROC curve for ' + names[i] + ' (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    #    plt.title('Receiver operating characteristic of lstm on patient level')
    plt.legend(loc="lower right", fontsize=fontsize - 3)
    plt.xticks(fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)

    if save_name is not None:
        plt.savefig(save_name + '.png')
    else:

        plt.show()

def output_metric_level(CMs, pred_sepsispatient_list, levels=[0.88], test_metric='specificity'):
    output = []

    prvs_now = []
    tprs_now = []
    tnrs_now = []

    for i in range(len(CMs)):
        tpr, tnr, prv = decompose_confusion(CMs[i])
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