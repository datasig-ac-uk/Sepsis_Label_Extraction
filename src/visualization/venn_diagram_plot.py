from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
import matplotlib.pyplot as plt
from src.visualization.plot_functions import decompose_confusion
from src.features.sepsis_mimic3_myfunction import *
from definitions import *
import seaborn as sns


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
    patient_icustay=df_pred.groupby('id')['rolling_hours'].max()
    patient_true_label = df_pred.groupby('id')['labels'].max()
    patient_pred_label = df_pred.groupby('id')['preds'].max()

    # get the predicted septic time and true septic time
    pred_septic_time = df_pred[df_pred['preds'] == 1].groupby('id')['rolling_hours'].min() - 1
    true_septic_time = df_pred[df_pred['labels'] == 1].groupby('id')['rolling_hours'].max() - 1
    ids = 0
    if cohort == 'correct_predicted':
        ids = df_pred[(df_pred['preds'] == 1) & (df_pred['labels'] == 1)]['id'].unique()
        df_pred1 = df_pred.loc[df_pred['id'].isin(ids)]
        pred_septic_time = df_pred1[df_pred1['preds'] == 1].groupby('id')['rolling_hours'].min()-1
        true_septic_time = df_pred1[df_pred1['labels'] == 1].groupby('id')['rolling_hours'].min() - 1 + T

    return patient_true_label, patient_pred_label.values, \
           confusion_matrix(patient_true_label,patient_pred_label), \
           pred_septic_time, true_septic_time, ids,patient_icustay


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

        _, patient_pred_label, CM, pred_septic_time, _, _,_ = patient_level_pred(df, labels_true, pred_labels, a1,
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
    Root_Data = DATA_processed + current_data + 'experiments_' + str(x) + '_' + str(y) + '/test/'
    Output_Data = OUTPUT_DIR + 'predictions/' + current_data
    thresholds = np.arange(precision) / precision
    models = ['LGBM', 'LSTM', 'CoxPHM']
    pred_sepsispatient_sublist = []
    pred_sepsispatient_sublist_list = []
    true_septic_perpatient_list = []

    for definition in definitions:
        print(definition)
        pred_sepsispatient_sublist = []
        path_df = DATA_DIR + '/raw/further_split/val_' + str(x) + '_' + str(y) + '.csv'
        df_sepsis1 = pd.read_pickle(Root_Data + definition[1:] + '_dataframe.pkl')
        current_label = np.load(Root_Data + 'label' + definition[1:] + '_' + str(T) + '.npy')
        for model in models:
            print(model)

            prob_preds = np.load(
                Output_Data + model + '/' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:] + '.npy')
            print(df_sepsis1.shape, current_label.shape, prob_preds.shape)
            pred_labels = (prob_preds > 0).astype('int')
            patient_true_label, _, _, _, _, _,_ = patient_level_pred(df_sepsis1, current_label, pred_labels, T,
                                                                   sample_ids=None, cohort='full')
            true_septic_perpatient_list.append(patient_true_label.values)
            CMs, patient_pred_label_list, _ = suboptimal_choice_patient(df_sepsis1, current_label,
                                                                        prob_preds, a1=T,
                                                                        thresholds=thresholds,
                                                                        sample_ids=None)
            patient_pred_label_at_levels, precision, tpr, tnr, _ = output_metric_level(CMs, patient_pred_label_list,
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


def sepsis_onset_time_plots(x, y, T, test_metric, metric_thresh, precision, save_dir,strict_exclusion=False):
    definitions = ['t_sofa', 't_suspicion', 't_sepsis_min']
    models = ['LGBM', 'LSTM', 'CoxPHM']
    current_data = 'blood_only_data/'
    Root_Data = DATA_processed + current_data + 'experiments_' + str(x) + '_' + str(y) + '/test/'
    Output_Data = OUTPUT_DIR + 'predictions/' + current_data
    thresholds = np.arange(precision) / precision

    if strict_exclusion:
        df_sepsis1 = pd.read_pickle(Root_Data + '_sepsis_min' + '_dataframe.pkl')
        sample_ids = df_sepsis1['icustay_id'].unique()
    else:
        sample_ids=None


    true_septic_time_list = []
    pred_septic_time_sublists = []
    ids_list = []
    for definition in definitions:
        df_sepsis1 = pd.read_pickle(Root_Data + definition[1:] + '_dataframe.pkl')
        current_label = np.load(Root_Data + 'label' + definition[1:] + '_' + str(T) + '.npy')
        for model in models:
            prob_preds = np.load(Output_Data + model + '/' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:] + '.npy')

            CMs, patient_pred_label_list, pred_septic_time_list = suboptimal_choice_patient(df_sepsis1, current_label,
                                                                                            prob_preds, \
                                                                                            a1=6, \
                                                                                            thresholds=thresholds,
                                                                                            sample_ids=sample_ids)
            _, _, _, _, idx = output_metric_level(CMs, patient_pred_label_list, levels=[metric_thresh],
                                                  test_metric=test_metric)
            threshold = thresholds[idx]
            # print(df_sepsis1.shape,current_labels.shape,pred_labels.shape,prob_preds.shape)
            pred_labels = (prob_preds > threshold).astype('int')
            patient_true_label, _, _, pred_septic_time, true_septic_time, ids,patient_icustay = patient_level_pred(df_sepsis1, current_label,
                                                                                  pred_labels, T, sample_ids=sample_ids,
                                                                                  cohort='full')
            ids_list.append(ids)
            true_septic_time_list.append(true_septic_time)
            pred_septic_time_sublists.append(pred_septic_time)
    true_id_list = [true_septic_time_list[i].index for i in range(len(true_septic_time_list))]
    identified_pred_sepsis_time = [pred_septic_time_sublists[i].loc[pred_septic_time_sublists[i].index.isin(true_id_list[i])]
                                   for i in range(len(pred_septic_time_sublists))]
    time_difference_dist_definitions(true_septic_time_list, identified_pred_sepsis_time,
                                     time_grid=[0, 6, 12, 18, 24, 30, 36, 42, 48],save_dir=save_dir)
    #time_difference_dist_model(true_septic_time_list, identified_pred_sepsis_time,
                               #time_grid=[0, 6, 12, 18, 24, 30, 36, 42, 48],save_dir=save_dir)
    median_time_difference(true_septic_time_list, identified_pred_sepsis_time, time_grid=np.arange(6, 16),save_dir=save_dir)
    #onset_time_dist_definitions(true_septic_time_list, identified_pred_sepsis_time,time_grid=np.arange(4, 16), save_dir=save_dir)
    #plt.figure(figsize=(8, 6))
    #labels=['H1','H2','H3']
    #for i in range(3):
     #   plt.hist(true_septic_time_list[3*i],label=labels[i],alpha=0.6,bins=50)
      #  plt.legend()
       # print('mean_'+labels[i],np.mean(np.array(true_septic_time_list[3*i])),'median_'+labels[i],np.median(np.array(true_septic_time_list[3*i])))

    true_onset_time_dist(true_septic_time_list)




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
            time_diff.append('>=' + str(time))
            models.append(model_grid[i])

            proportion.append(np.where(identified_time_difference > time)[0].shape[0] / len(true_septic_time_list[i]))
    data_dict = {'HBO': time_diff, 'model': models, 'proportion': proportion}
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    sns.barplot(x="HOB", y="proportion", hue="model", data=pd.DataFrame.from_dict(data_dict))
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
        identified_time_difference = identified_true_sepsis_time.values - identified_pred_sepsis_time[3 * i].values
        for time in time_grid:
            time_diff.append('>=' + str(time))
            defs.append(def_grid[i])

            proportion.append(
                np.where(identified_time_difference > time)[0].shape[0] / len(true_septic_time_list[3 * i]))
        print(len(true_septic_time_list[3 * i]))
    data_dict = {'HOB': time_diff, 'def': defs, 'proportion': proportion}
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    sns.barplot(x="HOB", y="proportion", hue="def", data=pd.DataFrame.from_dict(data_dict))
    plt.savefig(save_dir + 'Time_diff_dist_definitions' + '.png')


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
            icustay.append('>='+str(time))
            defs.append(def_grid[i])

            proportion.append(
                np.where(identified_true_sepsis_time >= time)[0].shape[0] / len(true_septic_time_list[3 * i]))
        print(len(true_septic_time_list[3 * i]))
    data_dict = {'icustay': icustay, 'def': defs, 'proportion': proportion}
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    sns.barplot(x='icustay', y="proportion", hue="def", data=pd.DataFrame.from_dict(data_dict))
    plt.savefig(save_dir + 'onset_time_dist_definitions' + '.png')


def true_onset_time_dist(true_septic_time_list,time_grid=np.arange(0,50)):
    def_grid = ['H1', 'H2', 'H3']
    icustay=[]
    proportion = []
    defs=[]
    for i in range(3):
        for time in time_grid:
            icustay.append( time)
            proportion.append(np.where(true_septic_time_list[3 * i]>=time)[0].shape[0])
            defs.append(def_grid[i])
    data_dict={ 'def': defs, 'number of septic patient': proportion,'icustay':icustay}
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    sns.lineplot(data=pd.DataFrame.from_dict(data_dict),x='icustay',y='number of septic patient',hue='def')




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
                    identified_true_sepsis_time.values[ids] - identified_pred_sepsis_time[3 * i + j].values[ids]))
                model_def.append(def_grid[i] + '&' + model_grid[j])
                icustay.append('>='+str(time))
    data_dict = {'icustay': icustay, 'model_definitions': model_def, 'median of HBO': median_list}
    df = pd.DataFrame.from_dict(data_dict)
    df['definition'] = [item[:2] for item in df['model_definitions'].values]
    df['model'] = [item[3:] for item in df['model_definitions'].values]
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    sns.lineplot(data=df, x='icustay', y='median of HBO', hue='definition', style='model', markers=True,sort=False)
    plt.legend(loc='upper left', prop={'size': 10})
    plt.savefig(save_dir + 'median of HOB' + '.png')

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
                identified_pred_sepsis_time[np.where(identified_true_sepsis_time==time)]

if __name__ == '__main__':
    plot_venn(24, 12, 6, 'sensitivity', 0.85, 2000, save_dir=OUTPUT_DIR + 'plots/')
    sepsis_onset_time_plots(24, 12, 6, 'sensitivity', 0.85, 2000, save_dir=OUTPUT_DIR + 'plots/')
