import sys
import time

import numpy as np
from ray.tune.utils import get_pinned_object
from ray import tune
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_curve, auc
import random
sys.path.insert(0, '../../')

from models.nets import LSTM
from data.torch_timeseries_dataset import LSTM_Dataset
import features.scaler as scaler
import constants
import omni.functions as omni_functions
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix,auc
def prepared_data_train(ts_dataset, labels, normalize, batch_size, device):
    dataset = ts_dataset
    if normalize:
        my_scaler = scaler.TrickScaler(scaling='mms').fit(dataset.data)
        dataset.data = my_scaler.transform(dataset.data)
    data = torch.FloatTensor(dataset.data.float())
    lengths = torch.FloatTensor(dataset.lengths)
    labels = torch.LongTensor(labels)
    ds = LSTM_Dataset(x=data, y=labels, p=20, lengths=lengths, device=device)
    dl = DataLoader(ds, batch_size=batch_size)
    return dl, my_scaler


def train_model(model, train_dl, n_epochs, save_dir, loss_func, optimizer):
    model.train()
    best_loss = 1e10
    for epoch in range(n_epochs):
        print(epoch)
        start = time.time()
        train_losses = []
        model.train()
        for step, (x, y) in enumerate(train_dl):
            optimizer.zero_grad()

            prediction = model(x)

            loss = loss_func(prediction, y.view(-1))

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print("train loss=", np.mean(train_losses))
        print('time=', time.time() - start)
        torch.save(model.state_dict(), save_dir)


def prepared_data_test(ts_dataset, labels, normalize, scaler, batch_size, device):
    if normalize:
        ts_dataset.data = scaler.transform(ts_dataset.data)
    data = torch.FloatTensor(ts_dataset.data.float())
    lengths = torch.FloatTensor(ts_dataset.lengths)
    labels = torch.LongTensor(labels)
    ds = LSTM_Dataset(x=data, y=labels, p=20, lengths=lengths, device=device)
    dl = DataLoader(ds, batch_size=batch_size)
    return dl


def eval_model(test_dl, model, save_dir):
    model.eval()
    test_preds, test_y = [], []
    with torch.no_grad():
        for step, (x, y) in enumerate(test_dl):
            test_y.append(y.view(-1))
            test_preds.append(torch.nn.functional.softmax(model(x), dim=1))

    def tfm(x): return torch.cat(x).cpu().detach().numpy()
    test_preds = tfm(test_preds)
    test_true = tfm(test_y)
    fpr, tpr, thresholds = roc_curve(
        test_true + 1, test_preds[:, 1], pos_label=2)
    index = np.where(tpr >= 0.85)[0][0]

    specificity = 1 - fpr[index]
    auc_score = auc(fpr, tpr)
    print(test_true.shape, test_preds.shape)
    print('auc and sepcificity for test ', auc_score, specificity)
    if save_dir is None:
        pass
    else:
        omni_functions._create_folder_if_not_exist(save_dir)
        np.save(save_dir, test_preds[:, 1])
    test_pred_labels = (test_preds[:, 1] > thresholds[index]).astype('int')
    tn, fp, fn, tp = confusion_matrix(test_true, test_pred_labels).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(test_true, test_pred_labels)
    print('accuracy=', accuracy)

    return auc_score, specificity,sensitivity, accuracy, test_true, test_preds[:, 1]

def eval_model1(test_dl, model,threshold, save_dir):
    model.eval()
    test_preds, test_y = [], []
    with torch.no_grad():
        for step, (x, y) in enumerate(test_dl):
            test_y.append(y.view(-1))
            test_preds.append(torch.nn.functional.softmax(model(x), dim=1))

    def tfm(x): return torch.cat(x).cpu().detach().numpy()

    prob_preds_test = tfm(test_preds)[:,1]
    test_labels = tfm(test_y)
    test_preds = np.array((prob_preds_test >= threshold).astype('int'))
    tn, fp, fn, tp = confusion_matrix(test_true, test_preds).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print('auc,sepcificity,sensitivity', roc_auc_score(
        test_labels, prob_preds_test), specificity, sensitivity)
    print('accuracy', accuracy_score(test_labels, test_preds))
    accuracy = accuracy_score(df['label'].values, test_preds)
    print('auc, sepcificity,accuracy', auc_score, specificity, accuracy)
    if save_dir is None:
        pass
    else:
        omni._create_folder_if_not_exist(filename)
        np.save(save_dir, prob_preds_test)

    return auc_score, specificity, sensitivity, accuracy, test_labels, prob_preds_test


def model_cv(config, data_list, device):
    ts_dataset, labels, train_patient_indices, train_full_indices, test_patient_indices, \
        test_full_indices, k = get_pinned_object(data_list)
    p, hidden_channels, linear_channels, epochs, seed = config["p"], config["hidden_channels"], \
        config["linear_channels"], config["epochs"],config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    test_true, test_preds = [], []
    data = torch.FloatTensor(ts_dataset.data.float())
    lengths = torch.FloatTensor(ts_dataset.lengths)
    labels = torch.LongTensor(labels)
    for i in range(k):
        train_idxs, test_idxs = train_full_indices[i], test_full_indices[i]
        train_id_idxs, test_id_idxs = train_patient_indices[i], test_patient_indices[i]
        # Make train and test data
        # split data into train and test
        train_lengths = lengths[train_id_idxs].to(device)
        train_data = data[train_id_idxs]
        train_labels = torch.cat([labels[i].to(device) for i in train_idxs])

        test_lengths = lengths[test_id_idxs].to(device)
        test_data = data[test_id_idxs]
        test_labels = torch.cat([labels[i].to(device) for i in test_idxs])

        # normalization
        my_scaler = scaler.TrickScaler(scaling='mms').fit(train_data)
        train_data = my_scaler.transform(train_data)
        test_data = my_scaler.transform(test_data)

        # parse the train and test data into LSTM torch dataset
        train_ds = LSTM_Dataset(
            x=train_data, y=train_labels, p=p, lengths=train_lengths, device=device)
        test_ds = LSTM_Dataset(x=test_data, y=test_labels,
                               p=p, lengths=test_lengths, device=device)
        del train_data, train_labels, train_lengths, test_lengths, test_data, test_labels
        torch.cuda.empty_cache()

        # Dataloaders.
        train_dl = DataLoader(train_ds, batch_size=128)
        test_dl = DataLoader(test_ds, batch_size=10000)

        model = LSTM(in_channels=data[0].size(1), num_layers=1, hidden_channels=hidden_channels,
                     hidden_1=linear_channels, out_channels=2,
                     dropout=0).to(device)

        train_model(model, train_dl, n_epochs=epochs,
                    save_dir=constants.MODELS_DIR +
                    '/LSTM_mimic' + 'cv_' + str(i),
                    loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=1e-3))
        model.load_state_dict(torch.load(constants.MODELS_DIR + '/LSTM_mimic' + 'cv_' + str(i),
                                         map_location=torch.device('cpu')))
        _, _, _, true, preds = eval_model(test_dl, model, None)
        test_true.append(true)
        test_preds.append(preds)
    test_true_full = np.concatenate(
        [test_true[i].reshape(-1, 1) for i in range(len(test_true))])
    test_preds_full = np.concatenate(test_preds)
    fpr, tpr, thresholds = roc_curve(
        test_true_full + 1, test_preds_full, pos_label=2)
    auc_score = auc(fpr, tpr)
    tune.report(mean_accuracy=auc_score)


search_space = {
    "p": tune.choice([20]),
    "hidden_channels": tune.choice([16, 32, 48, 64]),
    "linear_channels": tune.choice([16, 32, 48, 64]),
    "epochs": tune.sample_from(lambda _: np.random.randint(low=10, high=30)),
    "lr": tune.uniform(1e-4, 8e-4),
    "seed": 1234
}
