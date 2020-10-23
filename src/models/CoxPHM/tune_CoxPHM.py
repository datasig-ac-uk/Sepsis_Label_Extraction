from functools import partial

import torch
from src.features.scaler import *
import os
from definitions import *
from lifelines import CoxPHFitter
from src.models.CoxPHM.coxphm_functions import *
import ray
from ray import tune
from ray.tune.utils import pin_in_object_store, get_pinned_object
from src.features.sepsis_mimic3_myfunction import *

if __name__ == '__main__':
    current_data = 'blood_only_data/'
    signature=False
    Root_Data, Model_Dir, _ = folders(current_data, model='CoxPHM_no_sig')

    a1, a2, k = 6, 0, 5
    x, y = 24, 12

    Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/cv/'

    print(Data_Dir)

    for definition in ['t_sofa']:
        labels = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(a1) + '.npy')
        df = pd.read_pickle(Data_Dir + definition[1:] + '_dataframe.pkl')
        features = np.load(Data_Dir + 'james_features' + definition[1:] + '.npy')

        icustay_lengths = np.load(Data_Dir + 'icustay_lengths' + definition[1:] + '.npy')
        tra_patient_indices, tra_full_indices, val_patient_indices, val_full_indices = \
            cv_pack(icustay_lengths, k=k, definition=definition, path_save=Data_Dir, save=True)

        # prepare dataframe for coxph model
        df_coxph = Coxph_df(df, features, original_features, a1, labels,signature=False)
        ray.init(num_cpus=5)
        data = pin_in_object_store([df_coxph, tra_full_indices, val_full_indices, k])
        analysis = tune.run(partial(model_cv, data=data, a1=a1),
                            name='mimic_coxph' + definition[1:], config=search_space,
                            resources_per_trial={"cpu": 1}, num_samples=100,
                            max_failures=5, reuse_actors=True, verbose=1)
        best_trial = analysis.get_best_trial("mean_accuracy")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation auc: {}".format(
            best_trial.last_result["mean_accuracy"]))
        save_pickle(best_trial.config, Model_Dir + 'hyperparameter/' + 'config' + definition[1:])
        ray.shutdown()
