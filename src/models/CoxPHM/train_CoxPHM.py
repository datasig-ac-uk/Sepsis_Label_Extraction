import sys
import numpy as np
import pandas as pd


sys.path.insert(0, '../../')
import constants
import models.CoxPHM.coxphm_functions as coxphm_functions
import omni.functions as omni_functions
import features.mimic3_function as mimic3_myfunc



def train_CoxPHM(T_list, x_y, definitions, data_folder, signature, fake_test):
    """

    :param T_list(list of int): list of parameter T
    :param x_y(list of int):list of sensitivity parameter x and y
    :param definitions(list of str): list of definitions. e.g.['t_suspision','t_sofa','t_sepsis_min']
    :param data_folder(str): folder name specifying
    :return:
    """
    model = 'CoxPHM' if signature else 'CoxPHM_no_sig'
    config_dir = constants.MODELS_DIR + 'blood_only/CoxPHM/hyperparameter/config'
    data_folder = 'fake_test1/'+data_folder if fake_test else data_folder
    for x, y in x_y:

        Root_Data, Model_Dir, _, _ = mimic3_myfunc.folders(
            data_folder, model=model)
        Data_Dir = Root_Data + 'train' + '/'
        for definition in definitions:

            config = omni_functions.load_pickle(config_dir + definition[1:])
            print(definition, config)
            for T in T_list:
                print(definition, x, y, T)
                print('load dataframe and input features')
                df_sepsis_train = pd.read_pickle(
                    Data_Dir + str(x) + '_' + str(y) + definition[1:] + '_dataframe.pkl')
                features_train = np.load(
                    Data_Dir + 'james_features'+'_'+str(x) + '_' + str(y) + definition[1:] + '.npy')

                print('load train labels')
                labels_train = np.load(
                    Data_Dir + 'label' + '_'+str(x)+'_'+str(y)+'_'+str(T) + definition[1:] + '.npy')

                # prepare dataframe for coxph model
                df_coxph_train = coxphm_functions.Coxph_df(df_sepsis_train, features_train,
                                                           coxphm_functions.original_features, T, labels_train,
                                                           signature=signature)

                # fit CoxPHM
                cph = coxphm_functions.CoxPHFitter(penalizer=config['regularize']) \
                    .fit(df_coxph_train, duration_col='censor_hours', event_col='label',
                         show_progress=True, step_size=config['step_size'])

                omni_functions.save_pickle(
                    cph, Model_Dir + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])


if __name__ == '__main__':
    T_list = constants.T_list[1:2]
    data_folder = constants.exclusion_rules[0]
    x_y = constants.xy_pairs[:1]
    train_CoxPHM(T_list, x_y, constants.FEATURES,
                 data_folder, True, fake_test=False)
    #x_y = [(24, 12)]
    #data_folder_list = constants.exclusion_rules[1:]
    #for data_folder in data_folder_list:
       # train_CoxPHM(T_list, x_y, constants.FEATURES,
                     #data_folder, True, fake_test=False)
