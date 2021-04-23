from functools import partial
import os
import sys

import numpy as np
import ray
from ray import tune
from ray.tune.utils import pin_in_object_store
import torch

sys.path.insert(0, '../../')

import omni.functions as omni_functions
from data.dataset import TimeSeriesDataset
import features.mimic3_function as mimic3_myfunc
import models.LSTM.LSTM_functions as lstm_functions



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    current_data = 'blood_only/'
    Root_Data, Model_Dir, _, _ = mimic3_myfunc.folders(current_data, model='LSTM')
    print(Model_Dir)
    T, a2, k = 6, 0, 5
    x, y = 24, 12

    Data_Dir = Root_Data  +'train' + '/'
    definitions = ['t_sepsis_min']
    print(Data_Dir)

    for definition in definitions:
        labels = np.load(Data_Dir + 'label'+'_'+str(x)+'_'+str(y)+'_'+str(T) + definition[1:] + '.npy')
        dataset = TimeSeriesDataset().load(Data_Dir + str(x) + '_' + str(y) + definition[1:] + '_ffill.tsd')
        icustay_lengths = np.load(Data_Dir + 'icustay_lengths' + definition[1:] + '.npy')

        tra_patient_indices, tra_full_indices, val_patient_indices, val_full_indices = \
            mimic3_myfunc.cv_pack(icustay_lengths, k=k, definition=definition, path_save=Data_Dir, save=True)
        ray.init(num_gpus=4)
        data=pin_in_object_store([dataset, labels, tra_patient_indices,tra_full_indices,
                                                        val_patient_indices, val_full_indices,k])
        analysis = tune.run(partial(lstm_functions.model_cv, data_list=data, device=device),
                            name='mimic_lstm' + definition[1:], config=lstm_functions.search_space,
                            resources_per_trial={"gpu": 1}, num_samples=80,
                            max_failures=5, reuse_actors=True, verbose=1)
        best_trial = analysis.get_best_trial("mean_accuracy")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation auc: {}".format(
            best_trial.last_result["mean_accuracy"]))
        omni_functions.save_pickle(best_trial.config, Model_Dir + 'hyperparameter/' + 'config' + definition[1:])
        ray.shutdown()
