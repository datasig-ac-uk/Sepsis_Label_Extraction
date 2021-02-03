import sys

import numpy as np

sys.path.insert(0, '../')

import constants

import features.sepsis_mimic3_myfunction as mimic3_myfunc
import visualization.sepsis_mimic3_myfunction_patientlevel_clean as patientlevel_clean
import visualization.plot_functions_clean as plot_functions_clean

def auc_plots(definition_list,model_list,save_dir=constants.OUTPUT_DIR + 'plots/',T=6,train_test='test'):


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
                    probs_now = np.load(Output_Data + model + '/' + str(x) + '_' + str(y) + '_' + str(T) + definition[
                                                                                                   1:] + '_' + train_test + '.npy')
                else:
                    probs_now = np.load(Output_Data + model + '/' + 'prob_preds' + '_'+str(x) + '_' + str(y) + '_' + str(T) + '_' + definition[1:] + '.npy')

                icu_lengths_now = np.load(Data_Dir + 'icustay_lengths' + definition[1:] + '.npy')

                icustay_fullindices_now = mimic3_myfunc.patient_idx(icu_lengths_now)

                tpr, fpr = patientlevel_clean.patient_level_auc(labels_now, probs_now, icustay_fullindices_now, precision, n=n,a1=a1)

                labels_list.append(labels_now)
                probs_list.append(probs_now)
                tprs_list.append(tpr)
                fprs_list.append(fpr)

            names = ['48,24', '24,12', '12,6', '6,3']

            plot_functions_clean.auc_plot(labels_list, probs_list, names=names, \
                                  save_name=save_dir + 'auc_plot_instance_level_' + model + definition[
                                                                                            1:] + '_' + train_test)
            plot_functions_clean.auc_plot_patient_level(fprs_list, tprs_list, names=names, \
                                                save_name=save_dir + 'auc_plot_patient_level_' + model + definition[
                                                                                                         1:] + '_' + train_test)


if __name__ == '__main__':
    definition_list = [constants.FEATURES[1]]
    model_list = constants.MODELS
    auc_plots(definition_list=definition_list,model_list=model_list[1:],save_dir=constants.OUTPUT_DIR + 'plots/',T=6,train_test='train')

