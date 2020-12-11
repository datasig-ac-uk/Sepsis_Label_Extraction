import sys

from lightgbm import LGBMClassifier

sys.path.insert(0, '../../')
import constants
import features.sepsis_mimic3_myfunctionas as mimic3_myfunc
import models.LGBM.lgbm_functions as lgbm_func



if __name__ == '__main__':
        current_data='blood_culture_data/'
        Root_Data,Model_Dir,_,_,_= mimic3_myfunc.folders(current_data,model=constants.MODELS[0])

        a1,a2,k=6,0,5
        x,y=24,12
        n_iter=500
        
        Data_Dir=Root_Data+'experiments_'+str(x)+'_'+str(y)+'/cv/'

        print(Data_Dir)

        model=LGBMClassifier(random_state=42)
 
        for definition in constants.FEATURES:
            lgbm_func.feature_loading_model_tuning(model, Data_Dir,Model_Dir,definition,\
                                                   a1,lgbm_func.grid_parameters,n_iter=n_iter,k=k,save=True)
