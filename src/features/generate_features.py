import sys

sys.path.insert(0, '../')
import constants
import features.sepsis_mimic3_myfunction as sepsis_mimic3

if __name__ == '__main__':

    x, y = 24, 12
    a2 = 0

    path_df_train = '/scratch/mimiciii/training_data/further_split/train_{}_{}.csv'.format(x, y)
    path_df_test = '/scratch/mimiciii/training_data/further_split/val_{}_{}.csv'.format(x, y)

#     path_df_train = DATA_DIR + '/raw/full_culture_data/train_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'
#     path_df_test = DATA_DIR + '/raw/full_culture_data/test_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'


    Save_Dir_train = constants.DATA_processed + 'full_culture_data/experiments_{}_{}/train/'.format(x, y)
    Save_Dir_test =  constants.DATA_processed + 'full_culture_data/experiments_{}_{}/test/'.format(x, y)

    print('generate train/set features for sensitity {}_{} definition'.format(x, y))

    sepsis_mimic3.featureset_generator(path_df_train, Save_Dir_train, x=x, y=y, a2=a2, definitions=constants.FEATURES, T_list=constants.T_list)
    sepsis_mimic3.featureset_generator(path_df_test, Save_Dir_test, x=x, y=y, a2=a2, definitions=constants.FEATURES, T_list=constants.T_list)

