import numpy as np


import sys
sys.path.insert(0, '../../../')

from definitions import *
from src.features.scaler import *
from src.features.dicts import *
from src.omni.functions import *
from src.models.CoxPHM.coxphm_functions import *
from src.features.sepsis_mimic3_myfunction import *
from src.visualization.sepsis_mimic3_myfunction_patientlevel_clean import decompose_cms, output_at_metric_level
from src.visualization.venn_diagram_plot import *
if __name__ == '__main__':
    results = []
    results_patient_level=[]
    train_test='/test'
    T_list=[4,6,8,12]

    for x, y in [(6,3),(12,6),(24, 12),(48,24)]:
    
        current_data_folder = 'blood_only_data/'
        Root_Data,Model_Dir,Data_save=folders(current_data_folder,model='CoxPHM_no_sig')
    
        Data_Dir =Root_Data+'experiments_'+str(x)+'_'+str(y)+train_test+'/'
    


        for T in T_list:
            for definition in definitions:
                print('load dataframe and input features')
                df_sepsis = pd.read_pickle(Data_Dir + definition[1:] + '_dataframe.pkl')
                features = np.load(Data_Dir + 'james_features' + definition[1:] + '.npy')
            
                print('load test labels')
                labels = np.load(Data_Dir + 'label' + definition[1:] + '_' + str(T) + '.npy')

                # prepare dataframe for coxph model
                df_coxph = Coxph_df(df_sepsis, features, original_features, T, labels,signature=False)
                #print(df_coxph.shape)

                # load trained coxph model
                cph = load_pickle(Model_Dir + str(x) + '_' + str(y) + '_' + str(T) + definition[1:])

                #predict and evalute on test set
                create_folder(OUTPUT_DIR + 'predictions/' +current_data_folder+ 'CoxPHM_no_sig/')
                auc_score, specificity, accuracy = Coxph_eval(df_coxph, cph, T,
                                                           OUTPUT_DIR + 'predictions/' +current_data_folder+ 'CoxPHM_no_sig/'
                                                              +str(x) + '_' + str(y) + '_' + str(T)+
                                                              definition[1:]  + '.npy')
                preds = np.load(  OUTPUT_DIR + 'predictions/' +current_data_folder+ 'CoxPHM_no_sig/'
                                                              +str(x) + '_' + str(y) + '_' + str(T)+
                                                              definition[1:]  + '.npy')
                #CMs, _,_=suboptimal_choice_patient(df_sepsis, labels, preds, a1=6, thresholds=np.arange(10000) / 10000,
                 #                         sample_ids=None)
                #tprs, tnrs, fnrs, pres, accs = decompose_cms(CMs)

                #results_patient_level.append([str(x) + ',' + str(y), T,definition, "{:.3f}".format(auc(1 - tnrs, tprs)), \
                        #        "{:.3f}".format(output_at_metric_level(tnrs, tprs, metric_required=[0.85])), \
                          #      "{:.3f}".format(output_at_metric_level(accs, tprs, metric_required=[0.85]))])

                results.append([str(x) + ',' + str(y), T, definition, auc_score, specificity, accuracy])

        # save numerical results
    create_folder(OUTPUT_DIR + 'results/' + current_data_folder + 'CoxPHM_no_sig/')
   # results_patient_level_df = pd.DataFrame(results_patient_level, columns=['x,y', 'T','definition', 'auc', 'sepcificity', 'accuracy'])

   # results_patient_level_df.to_csv(OUTPUT_DIR + 'results/'+current_data_folder+ 'CoxPHM_no_sig'+train_test+'patient_level_results.csv')
    result_df = pd.DataFrame(results, columns=['x,y', 'T', 'definition', 'auc', 'speciticity', 'accuracy'])
    result_df.to_csv(OUTPUT_DIR + 'results/'+current_data_folder+ 'CoxPHM_no_sig'+train_test+'results.csv')