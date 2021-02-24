import sys

sys.path.insert(0, '../')
import constants
import features.sepsis_mimic3_myfunction as mimic3_myfunc
def features_wrapper(data_list, x_y_list, purpose):
    """

    Args:
        data_list (list): list of exlusion rules
        x_y_list (list): list of (x,y) values
        purpose (string) : 'train' or 'test'

    Returns:

    """    
    for current_data in data_list:
        root_dir, _, _, _ = mimic3_myfunc.folders(current_data)
        save_dir = root_dir + purpose + '/'
                    
        for x, y in x_y_list:
            if purpose == 'test':
                #TODO ask Lingyi to about the file name of the test data
                path_df = constants.MIMIC_DATA_DIRS[current_data] + '_08_10_20_sensitivity_' + str(x) + '_' + str(y) + '.csv'

            elif purpose == 'train':
                if x==48:
                    path_df = constants.MIMIC_DATA_DIRS[current_data] +  '_08_10_20.csv'
                else:
                    path_df = constants.MIMIC_DATA_DIRS[current_data] + '_08_10_20_sensitivity_' + str(x) + '_' + str(y) + '.csv'
            else:
                raise TypeError("purpose not recognised!")

            print('generate ' + purpose + ' features for sensitity {}_{} definition'.format(x, y))

            mimic3_myfunc.featureset_generator(path_df, save_dir, x=x, y=y, a2=0, definitions=constants.FEATURES,
                                               T_list=constants.T_list)



if __name__ == '__main__':
    #TODO ask yue to chekc if this feature_wrapper could work on her side
    data_list=['blood_only']
    features_wrapper(data_list,constants.xy_pairs, purpose='train')
    #other exclusion rules
    data_list=constants.exclusion_rules[1:]

    features_wrapper(data_list, x_y_list=[(24,12)], purpose='train')
    
#     current_data='blood_only_data/'
 
#     purpose='train' ### Otherwise 'train'
        
#     Root_Data, _, _,_ = mimic3_myfunc.folders(current_data)
    
#     Save_Dir =  Root_Data+purpose+'/'
    
#     a2=0
    
#     for x,y in constants.xy_pairs:

#         if purpose=='test':

#             path_df = '/scratch/mimiciii/training_data/further_split/val_{}_{}.csv'.format(x, y)
            
#         elif purpose=='train':
         
#         #path_df = '/scratch/mimiciii/training_data/further_split/train_{}_{}.csv'.format(x, y)          
#             if x!=48:
#                   path_df='/scratch/mimiciii/training_data/metavision_sepsis_blood_only_data_08_10_20_sensitivity_'+str(x)+'_'+str(y)+'.csv'
#             else:
#                   path_df='/scratch/mimiciii/training_data/metavision_sepsis_blood_only_data_08_10_20.csv'

#         else:
#             raise TypeError("purpose not recognised!")
 
#         print('generate '+purpose+' features for sensitity {}_{} definition'.format(x, y))
    
#         mimic3_myfunc.featureset_generator(path_df, Save_Dir, x=x, y=y, a2=a2, definitions=constants.FEATURES, T_list=constants.T_list)
