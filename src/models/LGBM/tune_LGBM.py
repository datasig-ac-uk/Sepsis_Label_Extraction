import sys
sys.path.insert(0, '../../')
import models.LGBM.LGBM_functions as lgbm_func
import features.mimic3_function as mimic3_myfunc
import constants
import pickle

from lightgbm import LGBMClassifier



if __name__ == '__main__':
    current_data = constants.exclusion_rules[0]

    Root_Data, Model_Dir, _, _ = mimic3_myfunc.folders(
        current_data, model=constants.MODELS[0])

    a1, a2, k = 6, 0, 5
    x, y = 24, 12
    n_iter, n_jobs = 500,8


    Data_Dir = Root_Data + 'train/'

    print(Data_Dir)

    model = LGBMClassifier(random_state=42,n_jobs=4)

    for definition in constants.FEATURES[:1]:
        lgbm_func.feature_loading_model_tuning(model, Data_Dir, Model_Dir, definition,
                                               a1, lgbm_func.grid_parameters, n_iter=n_iter,n_jobs=n_jobs, k=k, save=True)
        with open(Model_Dir + 'lgbm_best_paras' + definition[1:] + '.pkl', 'rb') as file:
            best_paras_ = pickle.load(file)
        print(best_paras_)


