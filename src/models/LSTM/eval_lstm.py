
import numpy as np
import os
import time
import torch
import sys
sys.path.insert(0, '../../../')

from definitions import *
from src.models.LSTM.lstm_functions import *
from src.features.sepsis_mimic3_myfunction import *

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    x, y = 24, 12
    current_data_folder='full_culture_data/'
    Root_Data,Model_Dir,Data_save=folders(current_data_folder)    
    Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/train/'

#     Save_Dir = DATA_DIR + '/processed/experiments_' + str(x) + '_' + str(y) + '/H3_subset/'

    results=[]
    for definition in definitions:
        config = load_pickle(Model_Dir+'hyperparameter/config'+definition[1:])
#         for T in [6]:
        for T in T_list:
            print('load timeseries dataset')
            dataset = TimeSeriesDataset().load(Data_Dir + definition[1:] + '_ffill.tsd')

            print('load train labels')
            labels_test = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')
            #get torch dataloader for lstm
            scaler=load_pickle(Model_Dir+'hyperparameter/scaler'+definition[1:])
#             scaler=load_pickle(MODELS_DIR+'/LSTM/hyperparameter/scaler'+definition[1:]+'H3_subset')
            test_dl=prepared_data_test(dataset,labels_test,True,scaler,1000,device)


            # specify torch model architecture and load trained model
            model = LSTM(in_channels=dataset.data.shape[-1], num_layers=1, hidden_channels=config['hidden_channels'],
                         hidden_1=config['linear_channels'], out_channels=2,
                         dropout=0).to(device)
            
            model.load_state_dict(torch.load(Model_Dir +definition[2:] +'_'+ str(x) + '_' + str(y) + '_' + str(T),
                           map_location=torch.device('cpu')))
#             model.load_state_dict(
#                 torch.load(MODELS_DIR + '/LSTM/H3_subset/' +definition[2:] +'_'+ str(x) + '_' + str(y) + '_' + str(T),
#                            map_location=torch.device('cpu')))
            #auc_score, specificity, accuracy = eval_model(test_dl,model,save_dir=OUTPUT_DIR + '/predictions/LSTM/' + 'lstm_' +
              #                                            definition[1:] + str(x)+'_'+str(y)+'_'+ str(T) +'.npy')
            auc_score, specificity, accuracy = eval_model(test_dl, model,
                                                          save_dir=None)
            results.append([str(x) + ',' + str(y), T, definition, auc_score, specificity, accuracy])
    # save numerical results
    result_df = pd.DataFrame(results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'accuracy'])
#     result_df.to_csv(OUTPUT_DIR + '/numerical_results/' + 'LSTM/' + str(x) + ',' + str(y) + "_results_H3_subset.csv")
    result_df.to_csv(Data_save+str(x)+ '_' +str(y)+'_lstm_test_results.csv')






