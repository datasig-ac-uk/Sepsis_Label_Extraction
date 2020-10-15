import numpy as np
import pandas as pd
import random

import os
import pickle

import sys
sys.path.insert(0, '../../')

from definitions import *

from src.features.sepsis_mimic3_myfunction import *
from src.visualization.sepsis_mimic3_myfunction_patientlevel_clean import *
from src.visualization.plot_functions_clean import *
from src.visualization.table_functions_clean import *

if __name__ == '__main__':
    
    current_data='blood_culture_data/'
    Root_Data,Model_Dir,Data_save=folders(current_data,model='LGBM')
    Data_save=Root_Data+'plots/'
    create_folder(Data_save)
    
    icu_number_pds=[]
    
    for x,y in xy_pairs:
        
        Data_Dir=Root_Data+'experiments_'+str(x)+'_'+str(y)+'/cv/'
        
        icu_number_pd=pd.read_csv(Data_Dir+'icu_number.csv')
        icu_number_pds.append(icu_number_pd)
        
    icu_numbers=pd.concat(icu_number_pds,0)
    
    
    septicdata1=np.array(icu_numbers[icu_numbers.definition=='t_sofa']['sepsis_no'])
    totaldata1=np.array(icu_numbers[icu_numbers.definition=='t_sofa']['total_icu_no'])
    data1=[septicdata1,totaldata1-septicdata1]

    septicdata2=np.array(icu_numbers[icu_numbers.definition=='t_suspicion']['sepsis_no'])
    totaldata2=np.array(icu_numbers[icu_numbers.definition=='t_suspicion']['total_icu_no'])
    data2=[septicdata2,totaldata2-septicdata2]

    septicdata3=np.array(icu_numbers[icu_numbers.definition=='t_sepsis_min']['sepsis_no'])
    totaldata3=np.array(icu_numbers[icu_numbers.definition=='t_sepsis_min']['total_icu_no'])
    data3=[septicdata3,totaldata3-septicdata3]

    label1=['Septic in H1','Non-septic in H1']
    label2=['Septic in H2','Non-septic in H2']
    label3=['Septic in H3','Non-septic in H3']

    labels=['48,24','24,12','12,6','6,3']
    xlabel='x,y'
    ylabel='Number of ICUstay'

    stacked_barplot3lists_compare(data1,data2, data3,\
                                  label1,label2,label3,\
                                  xlabel,ylabel,\
                                  labels,\
                                  width=0.4,\
                                  fontsize=16,\
                                  save_name=Data_save+'sepsis_ratio')
        