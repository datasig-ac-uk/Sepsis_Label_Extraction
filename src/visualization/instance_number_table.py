import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '../')
import constants
import features.sepsis_mimic3_myfunction as mimic3_myfunc

instances = []

current_data = 'blood_culture_data/'
Root_Data, Model_Dir, Data_save = mimic3_myfunc.folders(current_data, model='LGBM')
Data_save = Root_Data + 'summary/'
for x, y in constants.xy_pairs:

    a2, k = 0, 5

    Data_Dir = Root_Data + 'experiments_' + str(x) + '_' + str(y) + '/cv/'

    for a1 in constants.T_list:

        labels1 = np.load(Data_Dir + 'label' + constants.FEATURES[0][1:] + '_' + str(a1) + '.npy')
        labels2 = np.load(Data_Dir + 'label' + constants.FEATURES[1][1:] + '_' + str(a1) + '.npy')
        labels3 = np.load(Data_Dir + 'label' + constants.FEATURES[2][1:] + '_' + str(a1) + '.npy')

        if a1 == 12:
            #                 instances.append([str(x)+','+str(y),'-',str(len(labels1))+'&',\
            #                                   str(len(labels2))+'&',len(labels3)])
            instances.append([str(x) + ',' + str(y), '-', len(labels1), \
                              len(labels2), len(labels3)])
        sepsis_instance_number1 = len(np.where(labels1 == 1)[0])
        sepsis_instance_number2 = len(np.where(labels2 == 1)[0])
        sepsis_instance_number3 = len(np.where(labels3 == 1)[0])

        instances.append([str(x) + ',' + str(y), a1, sepsis_instance_number1, \
                          sepsis_instance_number2, sepsis_instance_number3])
#             instances.append([str(x)+','+str(y),a1,str(sepsis_instance_number1)+'&',\
#                               str(sepsis_instance_number2)+'&',sepsis_instance_number3])

instances_df = pd.DataFrame(instances, columns=['x,y', 'a1', 'instance' + constants.FEATURES[0][1:], \
                                                'instance' + constants.FEATURES[1][1:],
                                                'instance' + constants.FEATURES[2][1:]])

instances_df.to_csv(Data_save + 'instance_number(forwrite).csv')
