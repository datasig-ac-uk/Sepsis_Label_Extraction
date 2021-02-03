from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
import matplotlib.pyplot as plt
from src.visualization.sepsis_mimic3_myfunction_patientlevel_clean import decompose_confusion
import features.sepsis_mimic3_myfunction as mimic3_myfunc
import constants
import seaborn as sns
from visualization.plot_functions_clean import auc_plot

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

        _, patient_pred_label, CM, pred_septic_time, _, _, _ = patient_level_pred(df, labels_true, pred_labels, a1,
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
        tpr, tnr, prv,_,_ = decompose_confusion(CMs[i])
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


def sepsis_onset_time_plots(x, y, T, test_metric, metric_thresh, precision, save_dir, strict_exclusion=False):
    definitions = ['t_sofa', 't_suspicion', 't_sepsis_min']
    models = ['LGBM', 'LSTM', 'CoxPHM']
    current_data = 'blood_only_data/'
    Root_Data = constants.DATA_processed  + current_data + 'experiments_' + str(x) + '_' + str(y) + '/test/'
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
            df_sepsis1.loc[df_sepsis1['sepsis_hour'].notnull()].groupby('icustay_id').rolling_los_icu.max().astype(
                'int'))
        current_label = np.load(Root_Data + 'label' + definition[1:] + '_' + str(T) + '.npy')
        for model in models:
            prob_preds = np.load(
                Output_Data + model + '/' + str(x) + '_' + str(y) + '_' + str(T) + definition[1:]+'_test' + '.npy')

            CMs, patient_pred_label_list, pred_septic_time_list = suboptimal_choice_patient(df_sepsis1, current_label,
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
    time_difference_dist_definitions(true_septic_time_list, identified_pred_sepsis_time,
                                     time_grid=[0, 6, 12, 18, 24, 30, 36, 42, 48], save_dir=save_dir)

    #median_time_difference(true_septic_time_list, identified_pred_sepsis_time, time_grid=np.arange(6, 16),
              #             save_dir=save_dir)
    median_time_difference_interval(true_septic_time_list, identified_pred_sepsis_time,
                                    time_grid=[(4, 6), (6, 8), (8, 10), (10, 12),
                                               (12, 14), (14, 300)], save_dir=save_dir)
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
    proprotion_HBO_line_plot(patient_true_label_list, true_septic_time_list, identified_pred_sepsis_time,
                             patient_icustay_list,
                             time_grid=np.arange(61), save_dir=save_dir)



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

            proportion.append(np.where(identified_time_difference > time)[0].shape[0] / len(true_septic_time_list[i]))
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
        identified_time_difference = identified_true_sepsis_time.values - identified_pred_sepsis_time[3 * i].values
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

def median_time_difference_interval(true_septic_time_list, identified_pred_sepsis_time, time_grid=[(4,6),(6,8),(8,10),(10,12),
    (12,14),(14,1000)],save_dir=None):
    model_def = []
    median_list = []
    icustay = []
    def_grid = ['H1', 'H2', 'H3']
    model_grid = ['LGBM', 'LSTM', 'CoxPHM']
    for i in range(len(def_grid)):
        for j in range(len(model_grid)):
            identified_true_sepsis_time = true_septic_time_list[3 * i + j].loc[
                true_septic_time_list[3 * i + j].index.isin(identified_pred_sepsis_time[3 * i + j].index)]
            for k,time in enumerate(time_grid):
                ids = np.where((identified_true_sepsis_time.values >= time[0])&(identified_true_sepsis_time.values <= time[1]))[0]
                median_list.append(np.median(
                    identified_true_sepsis_time.values[ids] - identified_pred_sepsis_time[3 * i + j].values[ids]))
                model_def.append(def_grid[i] + '&' + model_grid[j])
                #icustay.append(r'['+str(time[0])+','+str(time[1])+r']') if k != len(time_grid)-1 \
                    #else icustay.append(r'['+str(time[0])+','+str(max(true_septic_time_list[3 * i + j]))+r']')
                icustay.append(r'[' + str(time[0]) + ',' + str(time[1]) + r']')
    data_dict = {'icustay': icustay, 'model_definitions': model_def, 'median of HBO': median_list}
    df = pd.DataFrame.from_dict(data_dict)
    df['definition'] = [item[:2] for item in df['model_definitions'].values]
    df['model'] = [item[3:] for item in df['model_definitions'].values]
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 17})
    sns.lineplot(data=df, x='icustay', y='median of HBO', hue='definition', style='model', markers=True,sort=False)
    plt.legend(loc='upper left', prop={'size': 17})
    plt.savefig(save_dir + 'median of HBO different icustays' + '.jpeg',dpi=350)



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
                    identified_true_sepsis_time.values[ids] - identified_pred_sepsis_time[3 * i + j].values[ids]))
                model_def.append(def_grid[i] + '&' + model_grid[j])
                icustay.append(r'$\geq$' + str(time))
    data_dict = {'icustay': icustay, 'model_definitions': model_def, 'median of HBO': median_list}
    df = pd.DataFrame.from_dict(data_dict)
    df['definition'] = [item[:2] for item in df['model_definitions'].values]
    df['model'] = [item[3:] for item in df['model_definitions'].values]
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 17})
    sns.lineplot(data=df, x='icustay', y='median of HBO', hue='definition', style='model', markers=True, sort=False)
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
    tp_icustay = true_septic_time_list[i].loc[true_septic_time_list[i].index.isin(identified_pred_sepsis_time[i].index)]
    tp_icustay_discharge = septic_los_list[i].loc[septic_los_list[i].index.isin(identified_pred_sepsis_time[i].index)]

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
            (np.where(tp_sepsis_time > time)[0].shape[0] + len(fn_ids) - np.where(fn_icustay < time)[0].shape[0]) / len(
                patient_true_label_list[i]))
        tn_proportion.append(
            (np.where(fp_sepsis_time > time)[0].shape[0] + len(tn_ids) - np.where(tn_icustay < time)[0].shape[0]) / len(
                patient_true_label_list[i]))
        tp_discharge_proportion.append(
            (np.where(tp_sepsis_time <= time)[0].shape[0] - np.where(tp_icustay_discharge < time)[0].shape[0]) / len(
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
                                     time_grid=np.arange(0,61), save_dir=None):
    defs = []
    proportion = []

    time_diff = []
    defs1 = []
    def_grid = ['H1', 'H2', 'H3']
    patience_def = ['flagged as septic', 'true septic']

    for i in range(3):
        identified_true_sepsis_time = true_septic_time_list[3 * i].loc[
            true_septic_time_list[3 * i].index.isin(identified_pred_sepsis_time[3 * i].index)]
        identified_time_difference = identified_true_sepsis_time.values - identified_pred_sepsis_time[3 * i].values
        fn_ids = [x for x in true_septic_time_list[3 * i].index if x not in identified_pred_sepsis_time[3 * i].index]
        fn_icustay = patient_icustay_list[3 * i].loc[fn_ids]
        negative_ids = [x for x in patient_true_label_list[3 * i].index if x not in true_septic_time_list[3 * i].index]
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
                  'proportion (septic patients in ICU / septic patients)': np.array(proportion), 'cohort': defs1}
    color_pal = sns.color_palette("colorblind", 6).as_hex()
    colors = ','.join(color_pal)
    # print(pd.DataFrame.from_dict(data_dict1))
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 17, 'legend.handlelength': 1.2})
    sns.lineplot(x="Hours before sepsis onset", y="proportion (septic patients in ICU / septic patients)", hue="def",
                 data=pd.DataFrame.from_dict(data_dict1), style='cohort', sort=False, color=colors)
    # plt.title('proportion of predicted septic and septic+ non_septic conditional on septic patients')
    locs, labels = plt.xticks()
    # ax.set_ylabel('proportion (patients in ICU/ total patients)')
    plt.xticks(np.arange(0, time_grid[-1] + 1, step=5))
    plt.savefig(save_dir + 'proprotion_HBO_line_plot' + '.jpeg', dpi=350)
    return pd.DataFrame.from_dict(data_dict1)



if __name__ == '__main__':
    #plot_venn(24, 12, 6, 'sensitivity', 0.85, 2000, save_dir=constants.OUTPUT_DIR + 'plots/')
    sepsis_onset_time_plots(24, 12, 6, 'sensitivity', 0.85, 2000, save_dir=constants.OUTPUT_DIR + 'plots/',strict_exclusion=False)
