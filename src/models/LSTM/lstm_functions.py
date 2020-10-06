import numpy as np
import torch
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report,roc_curve,auc
import time
from torch import nn, optim

import sys
sys.path.insert(0, '../../../')

from definitions import *
from src.features.scaler import *
from src.data.dataset import TimeSeriesDataset
from src.data.functions import torch_ffill
from src.data.torch_timeseries_dataset import LSTM_Dataset
from torch.utils.data import DataLoader, TensorDataset
from src.models.nets import LSTM
from src.features.sepsis_mimic3_myfunction import *


def prepared_data_train(ts_dataset,labels,normalize,batch_size,device):
    
    dataset=ts_dataset
    if normalize:
        scaler = TrickScaler(scaling='mms').fit(dataset.data)
        dataset.data = scaler.transform(dataset.data)
    data = torch.FloatTensor(dataset.data.float())
    lengths = torch.FloatTensor(dataset.lengths)
    labels = torch.LongTensor(labels)
    ds = LSTM_Dataset(x=data, y=labels, p=20, lengths=lengths, device=device)
    dl = DataLoader(ds, batch_size=batch_size)
    return dl,scaler

def train_model(model,train_dl,n_epochs,save_dir,loss_func,optimizer):
    model.train()
    best_loss=1e10
    for epoch in range(n_epochs):
        print(epoch)
        start = time.time()
        train_losses = []
        model.train()
        for step, (x,y) in enumerate(train_dl):
            optimizer.zero_grad()

            prediction = model(x)

            loss = loss_func(prediction, y.view(-1))

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print("train loss=", np.mean(train_losses))
        print('time=', time.time() - start)
        torch.save(model.state_dict(), save_dir)
        
        
def prepared_data_test(ts_dataset,labels,normalize,scaler,batch_size,device):

    if normalize:
        ts_dataset.data = scaler.transform(ts_dataset.data)
    data = torch.FloatTensor(ts_dataset.data.float())
    lengths = torch.FloatTensor(ts_dataset.lengths)
    labels = torch.LongTensor(labels)
    ds = LSTM_Dataset(x=data, y=labels, p=20, lengths=lengths, device=device)
    dl = DataLoader(ds, batch_size=batch_size)
    return dl


def eval_model(test_dl, model,save_dir):
    model.eval()
    test_preds, test_y = [], []
    with torch.no_grad():
        for step, (x, y) in enumerate(test_dl):
            test_y.append(y.view(-1))
            test_preds.append(torch.nn.functional.softmax(model(x), dim=1))

    tfm = lambda x: torch.cat(x).cpu().detach().numpy()
    test_preds = tfm(test_preds)
    test_true = tfm(test_y)
    fpr, tpr, thresholds = roc_curve(test_true + 1, test_preds[:, 1], pos_label=2)
    index = np.where(tpr >= 0.85)[0][0]
    specificity = 1 - fpr[index]
    auc_score = auc(fpr, tpr)
    print(test_true.shape, test_preds.shape)
    print('auc and sepcificity for test ', auc_score, specificity)
    if save_dir is None:
        pass
    else:
        np.save(save_dir, test_preds[:, 1])
    test_pred_labels = (test_preds[:, 1] > thresholds[index]).astype('int')
    accuracy = accuracy_score(test_true, test_pred_labels)
    print('accuracy=', accuracy)

    return auc_score, specificity, accuracy


