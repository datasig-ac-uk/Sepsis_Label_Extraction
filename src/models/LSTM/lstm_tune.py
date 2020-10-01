from src.features.sepsis_mimic3_myfunction import *

from src.data.dataset import TimeSeriesDataset

from src.features.scaler import *
from src.data.functions import torch_ffill
import os
from src.data.torch_timeseries_dataset import LSTM_Dataset
from torch.utils.data import DataLoader, TensorDataset
from src.models.train_eval_function import *
from src.models.nets import LSTM
from torch import nn, optim

import ray
from ray import tune
from ray.tune.utils import pin_in_object_store, get_pinned_object


def get_data(definition,a1,Data_Dir,path_df):

    # Data_Dir = './Aug_experiments_24_12/'
    # Data_Dir= './Aug_experiments_12_6/'
    print('sensitity 24_12 definition')


    df_sepsis1 = dataframe_from_definition_discard(path_df, definition=definition,a2=0)
    try:
        features = np.load(Data_Dir + 'james_features' + definition[1:] + '.npy')
    except:
        features = jamesfeature(df_sepsis1, Data_Dir=Data_Dir, definition=definition)
    print(features.shape)
    try:
        labels = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(a1) + '.npy')
    except:
        labels = label_generator(df_sepsis1, a1=a1, Data_Dir=Data_Dir, definition=definition, save=True)
        
    icustay_lengths, train_patient_indices, train_full_indices, test_patient_indices, test_full_indices = dataframe_cv_pack(
        df_sepsis1,
        k=5, definition=definition,
        path_save=Data_Dir, save=True)

    index = np.cumsum(np.array([0] + icustay_lengths))
    features_list = [torch.tensor(features[index[i]:index[i + 1]]) for i in range(index.shape[0] - 1)]
    column_list = [item for item in range(features.shape[1])]
    print('convert to timeseries dataset')
    try:
        dataset = TimeSeriesDataset().load(Data_Dir + '/processed/mimic' + definition[1:] + '_ffill.tsd')
    except:

        dataset = TimeSeriesDataset(data=features_list, columns=column_list, lengths=icustay_lengths)
        dataset.data = torch_ffill(dataset.data)
        dataset.data[torch.isnan(dataset.data)] = 0
        dataset.data[torch.isinf(dataset.data)] = 0
        dataset.save(Data_Dir + '/processed/mimic' + definition[1:] + '_ffill.tsd')

    print('processed ' + definition[1:] + 'data saved')

    normalize = True
    if normalize:
        scaler = TrickScaler(scaling='mms').fit(dataset.data)
        dataset.data = scaler.transform(dataset.data)
    data = torch.FloatTensor(dataset.data.float())
    lengths = torch.FloatTensor(dataset.lengths)
    labels = torch.LongTensor(labels)

    return data,lengths,labels,df_sepsis1,train_patient_indices,train_full_indices, test_patient_indices, test_full_indices

def cv(data,lengths,labels,train_patient_indices, train_full_indices, test_patient_indices, test_full_indices,k,**kwargs):
    test_true, test_preds=[],[]
    p=kwargs['p']
    hidden_channels = kwargs['hidden_channels']
    linear_channels = kwargs['linear_channels']
    for i in range(k):
        train_idxs, test_idxs = train_full_indices[i], test_full_indices[i]
        train_id_idxs, test_id_idxs = train_patient_indices[i], test_patient_indices[i]
        # Make train and test data
        # TODO: This should really be train/test/val.
        train_lengths = lengths[train_id_idxs].to(device)
        train_data = data[train_id_idxs]
        print(torch.isnan(train_data).any())
        train_labels = torch.cat([labels[i].to(device) for i in train_idxs])

        test_lengths = lengths[test_id_idxs].to(device)
        test_data = data[test_id_idxs]
        test_labels = torch.cat([labels[i].to(device) for i in test_idxs])
        train_ds = LSTM_Dataset(x=train_data, y=train_labels, p=p, lengths=train_lengths, device=device)
        test_ds = LSTM_Dataset(x=test_data, y=test_labels, p=p, lengths=test_lengths, device=device)
        del train_data, train_labels, train_lengths, test_lengths, test_data, test_labels
        torch.cuda.empty_cache()
        # Dataloaders. We use a batch size of 1 as we have lists not tensors.
        train_dl = DataLoader(train_ds, batch_size=128)
        test_dl = DataLoader(test_ds, batch_size=10000)

        model = LSTM(in_channels=data[0].size(1), num_layers=1, hidden_channels=hidden_channels, hidden_1=linear_channels, out_channels=2,
                     dropout=0).to(device)

        train_model(model, train_dl, test_dl, n_epochs=50, patience=5,
                    save_dir=MODELS_DIR + '/LSTM_mimic' + definition[1:] + 'cv_' + str(i),
                    loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=1e-3))
        model.load_state_dict(torch.load(MODELS_DIR + '/LSTM_mimic' + definition[1:] + 'cv_' + str(i),
                                         map_location=torch.device('cpu')))
        train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_auc, true, preds = eval_model(
            train_dl, test_dl, model)
        test_true.append(true)
        test_preds.append(preds)
        test_true_full = np.concatenate([test_true[i].reshape(-1, 1) for i in range(len(test_true))])
        test_preds_full = np.concatenate(test_preds)
        fpr, tpr, thresholds = roc_curve(test_true_full + 1, test_preds_full[:, 1], pos_label=2)
        auc_score=auc(fpr, tpr)

        return auc_score

def tune_model(config):
    p,hidden_channels,linear_channels = config["p"],config["hidden_channels"],config["linear_channels"]
    auc_score=cv(get_pinned_object(data), get_pinned_object(lengths), get_pinned_object(labels),
                 get_pinned_object(train_patient_indices),get_pinned_object(train_full_indices),
                 get_pinned_object(test_patient_indices),get_pinned_object(test_full_indices), k=5,
                 p=p,hidden_channels=hidden_channels,linear_channels=linear_channels)
    tune.report(accuracy=auc_score)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,4"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    ###Three different settings for x,y:
    path_df = DATA_DIR + '/raw/metavision_sepsis_data_10_09_20_sensitivity_24_12.csv'

    ##### For different x,y, set different Data_Dir
    model_name = 'lstm'
    Data_Dir = './Sep_experiments_24_12/'
    # Data_Dir = './Aug_experiments_24_12/'
    # Data_Dir= './Aug_experiments_12_6/'
    print('sensitity 24_12 definition')

    a1=6

    definition= 't_sofa'

    data, lengths, labels, df_sepsis1,train_patient_indices, train_full_indices,test_patient_indices, test_full_indices \
        = get_data(definition,a1,Data_Dir,path_df)
    ray.init(num_gpus=2,num_cpus=1)
    data,lengths,labels,df_sepsis1,train_patient_indices, train_full_indices, \
    test_patient_indices, test_full_indices = pin_in_object_store(data),\
                                     pin_in_object_store(lengths),\
                                     pin_in_object_store(labels),\
                                     pin_in_object_store(df_sepsis1),\
                                     pin_in_object_store(train_patient_indices),\
                                     pin_in_object_store(train_full_indices), \
                                     pin_in_object_store(test_patient_indices), \
                                     pin_in_object_store(test_full_indices)

    search_space = {
        "p":tune.choice([6, 8, 10,12,15,20,25]),
        "hidden_channels":tune.choice([32,64,128]),
        "linear_channels":tune.choice([16,32,64])

    }
    analysis = tune.run(tune_model, config=search_space, resources_per_trial= \
        {"gpu": 1, "cpu": 1}, num_samples=27, max_failures=5, reuse_actors=True)
    best_trial = analysis.get_best_trial("accuracy")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation auc: {}".format(
        best_trial.last_result["accuracy"]))


