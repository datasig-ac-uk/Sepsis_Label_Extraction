from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
import matplotlib.pyplot as plt
from src.visualization.plot_functions import output_metric_level
def patient_level_pred(df, true_labels, pred_labels, T, sample_ids=None):
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
    patient_true_label = df_pred.groupby('id')['labels'].max()
    patient_pred_label = df_pred.groupby('id')['preds'].max()

    # get the predicted septic time and true septic time
    pred_septic_time = df_pred[df_pred['preds'] == 1].groupby('id')['rolling_hours'].min()
    true_septic_time = df_pred[df_pred['labels'] == 1].groupby('id')['rolling_hours'].min() + T

    return patient_true_label.values, patient_pred_label.values, confusion_matrix(patient_true_label,
                                                                                  patient_pred_label), pred_septic_time, true_septic_time


def suboptimal_choice_patient(df, labels_true, prob_preds, a1=6, thresholds=np.arange(100)[1:-20] / 100, sample_ids=None):
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

        _, patient_pred_label, CM, pred_septic_time, _ = patient_level_pred(df, labels_true, pred_labels, a1,
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


def plot_venn(x, y, T, test_metric, metric_thresh, precision, cohort):
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
    thresholds = np.arange(precision) / precision
    Data_Dir = './further_split_{}_{}/'.format(x, y)
    models = ['lgbm_', 'lstm_', 'coxph_']
    pred_sepsispatient_sublist = []
    pred_sepsispatient_sublist_list = []
    true_septic_perpatient_list = []
    # compute patient id in H3 definition
    path_df = DATA_DIR + '/raw/further_split/val_' + str(x) + '_' + str(y) + '.csv'
    df_sepsis1 = dataframe_from_definition_discard(path_df, definition='t_sepsis_min', a2=0)
    sample_ids = df_sepsis1['icustay_id'].unique()
    print('number of patient by sepsis_min=', sample_ids.shape)
    for definition in definitions:
        print(definition)
        pred_sepsispatient_sublist = []
        path_df = DATA_DIR + '/raw/further_split/val_' + str(x) + '_' + str(y) + '.csv'
        df_sepsis1 = dataframe_from_definition_discard(path_df, definition=definition, a2=0)
        current_label = np.load(Data_Dir + 'label_test' + definition[1:] + '_' + str(T) + '.npy')
        for model in models:
            print(model)

            prob_preds = np.load(Data_Dir + model + 'prob_preds' + definition[1:] + '_' + str(T) + '.npy')
            print(df_sepsis1.shape, current_label.shape, prob_preds.shape)
            pred_labels = (prob_preds > 0).astype('int')
            patient_true_label, _, _, _, _ = patient_level_pred(df_sepsis1, current_label, pred_labels, T, sample_ids)
            true_septic_perpatient_list.append(patient_true_label)
            CMs, patient_pred_label_list,patient_septic_time_list = suboptimal_choice_patient(df_sepsis1, current_label,
                                                                                              prob_preds, a1=T,
                                                                                              thresholds=thresholds,
                                                                                              sample_ids=sample_ids)
            patient_pred_label_at_levels, precision, tpr, tnr = output_metric_level(CMs, patient_pred_label_list,
                                                                                    levels=[metric_thresh],
                                                                                    test_metric=test_metric)
            print(precision, tpr, tnr)
            pred_sepsispatient_sublist.append(patient_pred_label_at_levels)
        pred_sepsispatient_sublist_list.append(pred_sepsispatient_sublist)
    if cohort == 'full':
        sepsis_ids_min = np.where((true_septic_perpatient_list[-1] != 2))[0]
    elif cohort == 'sepsis':
        sepsis_ids_min = np.where((true_septic_perpatient_list[-1] == 1))[0]
    elif cohort == 'non_sepsis':
        sepsis_ids_min = np.where((true_septic_perpatient_list[-1] == 0))[0]
    # plot
    Definitions = ['LGBM', 'LSTM', 'CoxPHM']
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs = axs.ravel()
    for i in range(3):
        a = np.where(pred_sepsispatient_sublist_list[0][i][0][sepsis_ids_min] == 1)[0]
        b = np.where(pred_sepsispatient_sublist_list[1][i][0][sepsis_ids_min] == 1)[0]
        c = np.where(pred_sepsispatient_sublist_list[-1][i][0][sepsis_ids_min] == 1)[0]

        venn_values = venn_3counts(a, b, c)

        venn3(subsets=venn_values, set_labels=('H1', 'H2', 'H3'), alpha=0.5, ax=axs[i])

        axs[i].set_title(Definitions[i])
        # plt.show()
    plt.tight_layout()

if __name__ == '__main__':
    plot_venn(24,12,6,'sensitivity',0.85,2000,'full')
