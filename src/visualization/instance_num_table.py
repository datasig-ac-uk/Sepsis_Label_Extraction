import numpy as np
import pandas as pd
import sys

sys.path.insert(0, '../')
import features.mimic3_function as mimic3_myfunc
import constants as constants


instances=[]

current_data = constants.exclusion_rules[0]
Root_Data = constants.DATA_processed + current_data
Root_Data, _, Output_predictions, Output_results = mimic3_myfunc.folders(current_data, model=constants.MODELS[0])

purpose = 'test'
Data_Dir= Root_Data + purpose+'/'

print(Data_Dir)

for x,y in constants.xy_pairs:

    a2,k=0,5

    
    for a1 in constants.T_list:
       
  
            labels1=np.load(Data_Dir + 'label_' + str(x) + '_' + str(y) + '_' + str(a1) + constants.FEATURES[0][1:] + '.npy')
            labels2=np.load(Data_Dir + 'label_' + str(x) + '_' + str(y) + '_' + str(a1) + constants.FEATURES[1][1:] + '.npy')
            labels3=np.load(Data_Dir + 'label_' + str(x) + '_' + str(y) + '_' + str(a1) + constants.FEATURES[2][1:] + '.npy')

        
            if a1==12:
#                 instances.append([str(x)+','+str(y),'-',str(len(labels1))+'&',\
#                                   str(len(labels2))+'&',len(labels3)])
                instances.append([str(x)+','+str(y),'-',len(labels1),\
                                  len(labels2),len(labels3)])
            sepsis_instance_number1=len(np.where(labels1==1)[0])
            sepsis_instance_number2=len(np.where(labels2==1)[0])
            sepsis_instance_number3=len(np.where(labels3==1)[0])
            
            instances.append([str(x)+','+str(y),a1,sepsis_instance_number1,\
                              sepsis_instance_number2,sepsis_instance_number3])
#             instances.append([str(x)+','+str(y),a1,str(sepsis_instance_number1)+'&',\
#                               str(sepsis_instance_number2)+'&',sepsis_instance_number3])

instances_df = pd.DataFrame(instances, columns=['x,y','a1', 'instance'+constants.FEATURES[0][1:],\
                                               'instance'+constants.FEATURES[1][1:],'instance'+constants.FEATURES[2][1:]])

instances_df.to_csv(Data_Dir+'instance_number(forwrite).csv')
