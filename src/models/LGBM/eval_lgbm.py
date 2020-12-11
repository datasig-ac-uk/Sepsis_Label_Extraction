import pickle
import sys

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd


sys.path.insert(0, '../../')
import constants
import features.sepsis_mimic3_myfunction as mimic3_myfunc
import models.LGBM.lgbm_functions as lgbm_functions


if __name__ == '__main__':
    
    

    a2,k=0,5
    x,y=24,12

    current_data='blood_culture_data/'
    Root_Data,Model_Dir,_,Output_predictions,Output_results=mimic3_myfunc.folders(current_data,model=constants.MODELS[0])
    
    results=[]
    for x,y in constants.xy_pairs:
        Data_Dir_train=Root_Data+'experiments_'+str(x)+'_'+str(y)+'/train/'
        Data_Dir_test=Root_Data+'experiments_'+str(x)+'_'+str(y)+'/test/'
        
        
    
        for a1 in constants.T_list:
            
            for definition in constants.FEATURES:
        
                print(x,y,a1,definition)
            
                label_train=np.load(Data_Dir_train+'label'+definition[1:]+'_'+str(a1)+'.npy')
                feature_train=np.load(Data_Dir_train+'james_features'+definition[1:]+'.npy')

                label_test=np.load(Data_Dir_test+'label'+definition[1:]+'_'+str(a1)+'.npy')
                feature_test=np.load(Data_Dir_test+'james_features'+definition[1:]+'.npy')
        
                with open(Model_Dir+'lgbm_best_paras'+definition[1:]+'.pkl', 'rb') as file:
                    best_paras_=pickle.load(file)
                
                clf=LGBMClassifier(random_state=42).set_params(**best_paras_)
            
                _, prob_preds_test, auc,specificity,accuracy=lgbm_functions.model_training(clf,feature_train,feature_test,label_train,label_test)
        
                np.save(Output_predictions+'prob_preds_'+str(x)+'_'+str(y)+'_'+str(a1)+'_'+definition[1:]+'.npy',prob_preds_test)
                results.append([str(x)+','+str(y),a1,definition,auc,specificity,accuracy])
        
    result_df = pd.DataFrame(results, columns=['x,y','a1', 'definition', 'auc','speciticity','accuracy'])
    result_df.to_csv(Output_predictions+'lgbm_test_results.csv')
    
    mimic3_myfunc.main_result_tables(result_df,Output_results,model='lgbm',purpose='test')
        

 
