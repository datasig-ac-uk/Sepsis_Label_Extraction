import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc
import torch

sys.path.insert(0, '../../')
import constants
from data.dataset import TimeSeriesDataset
import omni.functions as omni_functions
import models.LSTM.lstm_functions as lstm_functions
from models.nets import LSTM
import features.sepsis_mimic3_myfunction as mimic3_myfunc
from visualization.sepsis_mimic3_myfunction_patientlevel_clean import decompose_cms, output_at_metric_level
import visualization.venn_diagram_plot as venn_diagram_plot

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    current_data_folder = 'absolute_values/'
    train_test = '/test'
    results = []
    results_patient_level=[]
    T_list=[6]
    for x, y in [(24,12)]:

        Root_Data, Model_Dir, Data_save = mimic3_myfunc.folders(current_data_folder,model='LSTM')
        Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + train_test+'/'

#     Save_Dir = DATA_DIR + '/processed/experiments_' + str(x) + '_' + str(y) + '/H3_subset/'



#         for T in [6]:
        for T in T_list:
            for definition in constants.FEATURES:
                config = omni_functions.load_pickle(constants.MODELS_DIR + 'hyperparameter/LSTM/config' + definition[1:])
                print('load timeseries dataset')
                dataset = TimeSeriesDataset().load(Data_Dir + definition[1:] + '_ffill.tsd')

                print('load train labels')
                labels_test = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')
                #get torch dataloader for lstm
                scaler = omni_functions.load_pickle(Model_Dir + 'hyperparameter/scaler' + definition[1:])
#               scaler=load_pickle(MODELS_DIR+'/LSTM/hyperparameter/scaler'+definition[1:]+'H3_subset')
                test_dl = lstm_functions.prepared_data_test(dataset, labels_test, True, scaler, 1000, device)


                 # specify torch model architecture and load trained model
                model = LSTM(in_channels=dataset.data.shape[-1], num_layers=1, hidden_channels=config['hidden_channels'],
                         hidden_1=config['linear_channels'], out_channels=2,
                         dropout=0).to(device)
            
                model.load_state_dict(torch.load(Model_Dir + '_' + str(x) + '_' + str(y) + '_' + str(T)+ definition[1:],
                           map_location=torch.device('cpu')))
                 #model.load_state_dict(
                 #torch.load(MODELS_DIR + '/LSTM/H3_subset/' +definition[2:] +'_'+ str(x) + '_' + str(y) + '_' + str(T),
                  #          map_location=torch.device('cpu')))
                mimic3_myfunc.create_folder(constants.OUTPUT_DIR + 'predictions/' + current_data_folder + 'LSTM/')
                auc_score, specificity, accuracy,true,preds = lstm_functions.eval_model(test_dl, model, save_dir=constants.OUTPUT_DIR + 'predictions/' +
                                                                                        current_data_folder+ 'LSTM/'
                                                                                        +str(x) + '_' + str(y) + '_' +
                                                                                        str(T) + definition[1:]  + '.npy')
                df_sepsis = pd.read_pickle(Data_Dir + definition[1:] + '_dataframe.pkl')
                CMs, _, _ = venn_diagram_plot.suboptimal_choice_patient(df_sepsis, true, preds, a1=6, thresholds=np.arange(200) / 200,
                                          sample_ids=None)
                tprs, tnrs, fnrs, pres, accs = decompose_cms(CMs)

                results_patient_level.append([str(x) + ',' + str(y), T,definition, "{:.3f}".format(auc(1 - tnrs, tprs)), \
                                "{:.3f}".format(output_at_metric_level(tnrs, tprs, metric_required=[0.85])), \
                                "{:.3f}".format(output_at_metric_level(accs, tprs, metric_required=[0.85]))])


            #auc_score, specificity, accuracy = eval_model(test_dl, model,
             #                                             save_dir=None)
                results.append([str(x) + ',' + str(y), T, definition, auc_score, specificity, accuracy])
             # save numerical results
    mimic3_myfunc.create_folder(constants.OUTPUT_DIR + 'results/' + current_data_folder + 'LSTM/')
    results_patient_level_df = pd.DataFrame(results_patient_level, columns=['x,y', 'T','definition', 'auc', 'sepcificity', 'accuracy'])

    results_patient_level_df.to_csv(constants.OUTPUT_DIR + 'results/'+current_data_folder+ 'LSTM'+train_test+'patient_level_results.csv')
    result_df = pd.DataFrame(results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'accuracy'])
    result_df.to_csv(constants.OUTPUT_DIR + 'results/'+current_data_folder+ 'LSTM'+train_test+'results.csv')






