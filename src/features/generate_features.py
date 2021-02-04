import sys

sys.path.insert(0, '../')
import constants
import features.sepsis_mimic3_myfunction as mimic3_myfunc
def features_wrapper(data_list,x_y_list,purpose,raw_dir_auto=False):
    """

    Args:
        data_list (list): list of exlusion rules
        x_y_list (list): list of (x,y) values
        purpose (string) : 'train' or 'test'

    Returns:

    """
    a2 = 0

    
    for current_data in data_list:
        Root_Data, _, _, _ = mimic3_myfunc.folders(current_data)
        Save_Dir = Root_Data + purpose + '/'
        
        strict_exlucion = True if current_data == 'strict_exclusion' else False
        data_folder1 = 'metavision_sepsis_blood_only_data' if strict_exlucion or current_data== 'blood_only' \
                else current_data
        
        raw_data_dir_manual='/scratch/mimiciii/training_data/' if strict_exlucion or current_data== 'blood_only'\
        else '/scratch/mimiciii/training_data/additional_experiments/'        
        raw_data_dir=constants.DATA_DIR + 'raw/'+purpose+'/' if raw_dir_auto else raw_data_dir_manual
        
        for x, y in x_y_list:


            if purpose == 'test':
                #TODO ask Lingyi to about the file name of the test data
                path_df = constants.DATA_DIR + 'raw/'+purpose+'/' + data_folder1 + '_08_10_20_sensitivity_' + \
                str(x) + '_' + str(y) + '.csv'

            elif purpose == 'train':

                # path_df = '/scratch/mimiciii/training_data/further_split/train_{}_{}.csv'.format(x, y)
                if x==48:
                    path_df=raw_data_dir + data_folder1+ '_08_10_20.csv'
                else:
                    path_df =raw_data_dir + data_folder1 + '_08_10_20_sensitivity_' + \
                str(x) + '_' + str(y) + '.csv'


            else:
                raise TypeError("purpose not recognised!")

            print('generate ' + purpose + ' features for sensitity {}_{} definition'.format(x, y))

            mimic3_myfunc.featureset_generator(path_df, Save_Dir, x=x, y=y, a2=a2, definitions=constants.FEATURES,
                                           T_list=constants.T_list)



if __name__ == '__main__':
    #TODO ask yue to chekc if this feature_wrapper could work on her side
    data_list=['blood_only']
    features_wrapper(data_list,constants.xy_pairs,purpose='train',raw_dir_auto=False)
    #other exclusion rules
    data_list=constants.exclusion_rules[1:]

    features_wrapper(data_list,[(24,12)],purpose='train')
    
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
