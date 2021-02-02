import sys

sys.path.insert(0, '../')
import constants
import features.sepsis_mimic3_myfunction as mimic3_myfunc

if __name__ == '__main__':
   
    
    current_data='blood_culture_data/'
 
    purpose='train' ### Otherwise 'train'
        
    Root_Data, _, _,_ = mimic3_myfunc.folders(current_data)
    
    Save_Dir =  Root_Data+purpose+'/'
    
    a2=0
    
    for x,y in constants.xy_pairs:

        if purpose=='test':
            
            path_df = '/scratch/mimiciii/training_data/further_split/val_{}_{}.csv'.format(x, y)  
            
        elif purpose=='train':
         
        #path_df = '/scratch/mimiciii/training_data/further_split/train_{}_{}.csv'.format(x, y)          
            if x!=48:
                  path_df_cv='/scratch/mimiciii/training_data/metavision_sepsis_blood_only_data_08_10_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'
            else:
                  path_df_cv='/scratch/mimiciii/training_data/metavision_sepsis_blood_only_data_08_10_20.csv'

        else:
            raise TypeError("purpose not recognised!")
 
        print('generate '+purpose+' features for sensitity {}_{} definition'.format(x, y))
    
        mimic3_myfunc.featureset_generator(path_df, Save_Dir, x=x, y=y, a2=a2, definitions=constants.FEATURES, T_list=constants.T_list)


"""
    #Hang's feature generation
    a2 = 0
    # generate features for all x,y and T  for blood only data
    xy_pairs = [(24, 12)]
    data_folder = 'no_gcs'
    fake_test = False

    for x, y in xy_pairs:
        path_df_train = constants.DATA_DIR + 'raw/further_split/train_' + str(x) + '_' + str(y) + '.csv' if fake_test else \
            constants.DATA_DIR + 'raw/no_gcs_08_10_20_sensitivity_' + str(x) + '_' + str(y) + '.csv'
            #DATA_DIR + 'raw/metavision_sepsis_blood_only_data_08_10_20_sensitivity_' + str(x) + '_' + str(y) + '.csv'

        path_df_test = constants.DATA_DIR + 'raw/further_split/val_' + str(x) + '_' + str(y) + '.csv' if fake_test else \
            constants.DATA_DIR + 'raw/no_gcs_08_10_20_sensitivity_' + str(x) + '_' + str(y) + '.csv'

        #      path_df_train = DATA_DIR + '/raw/full_culture_data/train_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'
        #     path_df_test = DATA_DIR + '/raw/full_culture_data/test_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'

        Save_Dir_train = constants.DATA_processed + 'fake_test1/' + data_folder + '/experiments_' + str(x) + '_' + str(
            y) + '/train/' if fake_test \
            else constants.DATA_processed + data_folder + '/experiments_' + str(x) + '_' + str(y) + '/train/'
        Save_Dir_test = constants.DATA_processed + 'fake_test1/' + data_folder + '/experiments_' + str(x) + '_' + str(
            y) + '/test/' if fake_test \
            else constants.DATA_processed + data_folder + '/experiments_' + str(x) + '_' + str(y) + '/test/'

        print('generate train/set features for sensitity ' + str(x) + '_' + str(y) + ' definition')

        sepsis_mimic3.featureset_generator(path_df_train, Save_Dir_train, x=x, y=y, a2=a2, definitions=constants.FEATURES, T_list=constants.T_list, strict_exclusion=False)
        sepsis_mimic3.featureset_generator(path_df_test, Save_Dir_test, x=x, y=y, a2=a2, definitions=constants.FEATURES, T_list=constants.T_list, strict_exclusion=False)

    # generate features for x,y =24,12 and different data
    x, y = 24, 12
    data_list = ['no_gcs_']
    for data_folder in data_list:
        strict_exlucion = True if data_folder == 'strict_exclusion' else False
        if fake_test:
            data_folder1 = '' if strict_exlucion else data_folder
        else:
            data_folder1 = 'metavision_sepsis_blood_only_data_' if strict_exlucion else data_folder
        path_df_train = DATA_DIR + 'raw/further_split/' + data_folder1 + 'train_' + str(x) + '_' + str(
            y) + '.csv' if fake_test else \
            DATA_DIR + 'raw/' + data_folder1 + '08_10_20_sensitivity_' + \
            str(x) + '_' + str(y) + '.csv'
        path_df_test = DATA_DIR + 'raw/further_split/' + data_folder1 + 'val_' + str(x) + '_' + str(
            y) + '.csv' if fake_test else \
            DATA_DIR + 'raw/' + data_folder1 + '08_10_20_sensitivity_' + \
            str(x) + '_' + str(y) + '.csv'

        #     path_df_train = DATA_DIR + '/raw/full_culture_data/train_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'
        #     path_df_test = DATA_DIR + '/raw/full_culture_data/test_data/metavision_sepsis_data_18_09_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'

        Save_Dir_train = DATA_processed + 'fake_test1/' + data_folder + '/experiments_' + str(x) + '_' + str(
            y) + '/train/' if fake_test \
                else DATA_processed + data_folder + '/experiments_' + str(x) + '_' + str(y) + '/train/'

        Save_Dir_test = DATA_processed + 'fake_test1/' + data_folder + '/experiments_' + str(x) + '_' + str(
            y) + '/test/' if fake_test \
                else DATA_processed + data_folder + '/experiments_' + str(x) + '_' + str(y) + '/test/'

        print('generate train/set features for sensitity ' + str(x) + '_' + str(y) + ' definition')

        featureset_generator(path_df_train, Save_Dir_train, x=x, y=y, a2=a2, definitions=definitions, T_list=T_list,
                             strict_exclusion=strict_exlucion)
        featureset_generator(path_df_test, Save_Dir_test, x=x, y=y, a2=a2, definitions=definitions, T_list=T_list,
                             strict_exclusion=strict_exlucion)
"""
