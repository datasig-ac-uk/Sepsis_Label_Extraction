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
from src.models.optimizers import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(os.environ["CUDA_VISIBLE_DEVICES"])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
###Three different settings for x,y:
path_df = DATA_DIR + '/raw/metavision_sepsis_data_10_09_20_sensitivity_24_12.csv'
#path_df = DATA_DIR + '/raw/metavision_sepsis_blood_only_data_11_09_20_sensitivity_24_12.csv'
#path_df = DATA_DIR + '/raw/metavision_sepsis_data_13_08_20_sensitivity_12_6.csv'
#path_df = DATA_DIR + '/raw/metavision_sepsis_data_13_08_20_sensitivity_24_12.csv'


##### For different x,y, set different Data_Dir
model_name='lstm'
Data_Dir = DATA_DIR+'/processed/Sep_experiments_24_12/'
#Data_Dir = './Aug_experiments_24_12/'
#Data_Dir= './Aug_experiments_12_6/'
print('sensitity 24_12 definition')
# define definition 't_sofa','t_suspicion', 't_sepsis_min'
results=[]
for a1 in [4,6,8,12]:
    print('a1=',a1)
    for definition in ['t_suspicion', 't_sofa', 't_sepsis_min']:
        print(definition)


        df_sepsis1 = dataframe_from_definition_discard(path_df,definition=definition)

        try:
            features = np.load(Data_Dir+'james_features' + definition[1:] + '.npy')
        except:
            features = jamesfeature(df_sepsis1,Data_Dir=Data_Dir, definition=definition)
        print(features.shape)
        try:
            scores = np.load(Data_Dir+'scores'+definition[1:]+'_'+str(a1)+'.npy')
            labels = np.load(Data_Dir+'label'+definition[1:]+'_'+str(a1)+'.npy')
        except:
            labels, scores = label_scores(df_sepsis1, a1=a1, Data_Dir=Data_Dir,definition=definition, save=True)
        icustay_lengths, train_patient_indices, train_full_indices, test_patient_indices, test_full_indices = dataframe_cv_pack(
        df_sepsis1,
        k=5, definition=definition,
        path_save=Data_Dir,save=True)

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

        accuracy, precision, recall, f1_score, auc_score, test_true, test_preds, utility, pred_label = [], [], [], [], [], [], [], [], []
        for i in range(5):
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
            test_scores = np.concatenate([scores[i] for i in test_idxs])
            train_ds = LSTM_Dataset(x=train_data, y=train_labels, p=20, lengths=train_lengths, device=device)
            test_ds = LSTM_Dataset(x=test_data, y=test_labels, p=20, lengths=test_lengths, device=device)
            del train_data, train_labels, train_lengths, test_lengths, test_data, test_labels
            torch.cuda.empty_cache()
            # Dataloaders. We use a batch size of 1 as we have lists not tensors.
            train_dl = DataLoader(train_ds, batch_size=128)
            test_dl = DataLoader(test_ds, batch_size=10000)

            model = LSTM(in_channels=data[0].size(1), num_layers=1, hidden_channels=50, hidden_1=30, out_channels=2,
                     dropout=0).to(device)

            train_model(model, train_dl, test_dl, n_epochs=50, patience=5,
                    save_dir=MODELS_DIR + '/LSTM_mimic' + definition[1:] + 'cv_' + str(i),
                    loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=1e-3))
            model.load_state_dict( torch.load(MODELS_DIR + '/LSTM_mimic' + definition[1:] + 'cv_' + str(i),
                                          map_location=torch.device('cpu')))
            train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_auc, true, preds = eval_model(
                train_dl, test_dl, model)
            print(preds.shape, test_scores.shape)
            thresh = optimize_utility_threshold(preds[:, 1], scores=test_scores)
            print("Threshold now:", thresh)
            test_utility = compute_utility(test_scores, preds[:, 1], thresh)
            print('Utility score for : {:.3f}'.format(test_utility))
            thresh = optimize_f1_threshold(preds, true, budget=200, num_workers=1)
            print('F1 score threshold=', thresh)
            test_pred_label = (preds[:, 1] > 0.5).astype('int')
            print(test_pred_label.shape)
            print(true.shape)
            utility.append(test_utility)
            accuracy.append(test_accuracy)
            precision.append(test_precision)
            recall.append(test_recall)
            f1_score.append(test_f1_score)
            auc_score.append(test_auc)
            test_true.append(true)
            test_preds.append(preds)
            pred_label.append(test_pred_label)
        print('t_' + definition[1:] + 'definition:')
        print('test_mean_accuracy=', np.mean(np.array(accuracy)))
        print('test_mean_precision=', np.mean(np.array(precision)))
        print('test_mean_recall=', np.mean(np.array(recall)))
        print('test_mean_f1=', np.mean(np.array(f1_score)))
        print('test_mean_auc=', np.mean(np.array(auc_score)))
        print('test_mean_utility=', np.mean(np.array(test_utility)))
        test_true_full = np.concatenate([test_true[i].reshape(-1, 1) for i in range(len(test_true))])
        test_preds_full = np.concatenate(test_preds)
        test_pred_labels = np.concatenate([pred_label[i].reshape(-1, 1) for i in range(len(pred_label))])
        print(test_true_full.shape, test_pred_labels.shape)
        print('Inference on definition t_sepsis = ', definition[1:])
        fpr, tpr, thresholds = roc_curve(test_true_full + 1, test_preds_full[:, 1], pos_label=2)
        index=np.where(tpr>=0.85)[0][0]
        specificity=1-fpr[index]
        print('auc and sepcificity',auc(fpr, tpr),specificity)
        print('threshold for fixed 0.85 sensitity level is :',thresholds[index])
        np.save(Data_Dir +'lstm_'+ 'prob_preds' + definition[1:] + '_' + str(a1) + '.npy', test_preds_full[:,1])
        test_pred_labels= (test_preds_full[:, 1] > thresholds[index]).astype('int')
        precision, recall, f1_score, support = precision_recall_fscore_support(test_true_full, test_pred_labels,
                                                                       average='weighted')
        accuracy=accuracy_score(test_true_full, test_pred_labels)
        print("train_accuracy=", train_accuracy,
              "test_accuracy=", test_accuracy,
              "test_precision=", precision,
              "test_recall=", recall,
              "test_f1_score=", f1_score,
              "test_auc_score=", auc(fpr, tpr))
        print(classification_report(test_true_full, test_pred_labels, digits=4))
        print(confusion_matrix(test_true_full, test_pred_labels))
        results.append(['24,12', a1, definition, auc(fpr,tpr), specificity, accuracy])

result_df = pd.DataFrame(results, columns=['x,y','a1', 'definition', 'auc','speciticity','accuracy'])
result_df.to_csv(Data_Dir+model_name+"_24_12_Sep_results.csv")