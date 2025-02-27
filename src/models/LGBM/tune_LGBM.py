from lightgbm import LGBMClassifier

import constants
import features.mimic3_function as mimic3_myfunc
import models.LGBM.LGBM_functions as lgbm_func


if __name__ == '__main__':
    current_data = constants.exclusion_rules[0]

    Root_Data, Model_Dir, _, _ = mimic3_myfunc.folders(
        current_data, model=constants.MODELS[0])

    a1, a2, k = 6, 0, 5
    x, y = 24, 12
    n_iter, n_jobs = 500, 8

    Data_Dir = Root_Data + 'test/'

    print(Data_Dir)

    model = LGBMClassifier(random_state=42, n_jobs=constants.N_CPUS)

    for definition in constants.FEATURES:
        lgbm_func.feature_loading_model_tuning(model, Data_Dir, Model_Dir, definition,
                                               a1, lgbm_func.grid_parameters, n_iter=n_iter, k=k,
                                               n_jobs=constants.N_CPUS, save=True)
