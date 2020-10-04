import numpy as np
import pandas as pd
import random
import os
import pickle
from scipy.stats import iqr

import sys
sys.path.insert(0, '../../')
from definitions import *
from src.features.sepsis_mimic3_myfunction import *


def statistics_gender_mortality(df,variable):
    
 
    variable_=np.array(df.groupby(by='icustay_id').apply(lambda x:x[variable].iloc[0]))
        
    return len(np.where(variable_==1.0)[0])/len(variable_)

def statistics_others(df,variable):

    variable_=np.array(df.groupby(by='icustay_id').apply(lambda x:x[variable].iloc[0]))
        
    return (np.median(variable_),iqr(variable_))


def statistics_groups(dfs,variable,function=statistics_gender_mortality):
    
    output=[]
    num=len(dfs)
    
    for i in range(num):
        output.append(function(dfs[i],variable))
        
    return output



if __name__ == '__main__':

    
    print("Give dataset summary: gender,mortality,age,los, sepsis hour since ICU.")
    Root_Data=DATA_processed+'full_culture_data/'
    Data_save=Root_Data+'summary/'        
    create_folder(Data_save)
    
    a1,a2=6,0
    genders=[]
    mortalities=[]
    ages=[]
    los=[]
    sepsis_hours=[]

    for x,y in xy_pairs[1:2]:

        if x!=48:
            df_path='/scratch/mimiciii/training_data/further_split/val_'+str(x)+'_'+str(y)+'.csv'

        else:
            df_path='/data/raw/training_data/metavision_sepsis_data_18_09_20.csv'

        genders_=[]
        mortalities_=[]
        ages_=[]
        los_=[]
        sepsis_hours_=[]

        for definition in definitions:
        
            print(x,y, definition)
            df_now=dataframe_from_definition_discard(df_path,a1=a1,a2=a2,definition=definition)
        
        
            df_non=df_now[np.isnan(df_now.sepsis_hour)]
            df_sep=df_now[~np.isnan(df_now.sepsis_hour)]
        
            dfs=[df_now,df_non,df_sep]
            [gender,gender_non,gender_sep]=statistics_groups(dfs,'gender',function=statistics_gender_mortality)
            [mort,mort_non,mort_sep]=statistics_groups(dfs,'mortality',function=statistics_gender_mortality)

    
            if definition=='t_sofa':
                genders_.append([str(x)+','+str(y), definition,"{:.2%}".format(gender),\
                             "{:.2%}".format(gender_non), "{:.2%}".format(gender_sep)])
                mortalities_.append([str(x)+','+str(y), definition, "{:.2%}".format(mort),\
                                 "{:.2%}".format(mort_non), "{:.2%}".format(mort_sep)])
            
            else:
                genders_.append([str(x)+','+str(y), definition,"{:.2%}".format(gender),\
                             '-', "{:.2%}".format(gender_sep)])
                mortalities_.append([str(x)+','+str(y), definition,"{:.2%}".format(mort),\
                             '-', "{:.2%}".format(mort_sep)])    
        
            [(median_age,iqr_age),(median_age_non,iqr_age_non),(median_age_sep,iqr_age_sep)]=\
                                statistics_groups(dfs,'age',function=statistics_others)
            [(median_los,iqr_los),(median_los_non,iqr_los_non),(median_los_sep,iqr_los_sep)]=\
                                statistics_groups(dfs,'rolling_los_icu',function=statistics_others)
                
            if definition=='t_sofa':
                ages_.append([str(x)+','+str(y), definition, "{:.2f}".format(median_age)+' \pm '+"{:.2f}".format(iqr_age),\
                                          "{:.2f}".format(median_age_non)+' \pm '+"{:.2f}".format(iqr_age_non),\
                                          "{:.2f}".format(median_age_sep)+' \pm '+"{:.2f}".format(iqr_age_sep)])
                los_.append([str(x)+','+str(y), definition, "{:.2f}".format(median_los)+' \pm'+"{:.2f}".format(iqr_los),\
                                          "{:.2f}".format(median_los_non)+' \pm '+"{:.2f}".format(iqr_los_non),\
                                          "{:.2f}".format(median_los_sep)+' \pm '+"{:.2f}".format(iqr_los_sep)])
            else:
                ages_.append([str(x)+','+str(y), definition, "{:.2f}".format(median_age)+' \pm'+"{:.2f}".format(iqr_age),\
                                          '-',\
                                          "{:.2f}".format(median_age_sep)+' \pm '+"{:.2f}".format(iqr_age_sep)])
                los_.append([str(x)+','+str(y), definition, "{:.2f}".format(median_los)+' \pm'+"{:.2f}".format(iqr_los),\
                                          '-',\
                                          "{:.2f}".format(median_los_sep)+' \pm '+"{:.2f}".format(iqr_los_sep)])    
    

            (median_hours_sep,iqr_hours_sep)=statistics_others(df_sep,'sepsis_hour')
    
            sepsis_hours_.append([str(x)+','+str(y), definition, \
                             "{:.2%}".format(median_hours_sep)+' \pm '+"{:.2f}".format(iqr_hours_sep),\
                              '-',  '-'])
    
            print('\n')
    
        gender_df = pd.DataFrame(genders_,columns=['x,y', 'definition', 'gender','gender_nonsepsis','gender_sepsis'])                                        
        mort_df = pd.DataFrame(mortalities_,columns=['x,y', 'definition', 'mortality','mortality_nonsepsis','mortality_sepsis'])
        age_df = pd.DataFrame(ages_, columns=['x,y', 'definition','median_age \pm iqr_age',\
                                            'median_age_nonsepsis\pm iqr_age_nonsepsis',\
                                             'median_age_sepsis\pm iqr_age_sepsis'])

        los_df = pd.DataFrame(los_, columns=['x,y', 'definition','median_los\pm iqr_los',\
                                             'median_los_nonsepsis\pm iqr_los_nonsepsis',\
                                             'median_los_sepsis\pm iqr_los_sepsis'])

        sepsis_hours_df = pd.DataFrame(sepsis_hours_,\
                                    columns=['x,y', 'definition','median_sepsis_hour \pm iqr_sepsis_hour','-','-'])

        genders.append(gender_df)
        mortalities.append(mort_df)
        ages.append(age_df)
        los.append(los_df)
        sepsis_hours.append(sepsis_hours_df)
    
    print("Save dataset summary to",Data_save)
    
    gender_df=pd.concat(genders,0) 
    gender_df.to_csv(Data_save+'gender.csv')
    
    mortality_df=pd.concat(mortalities,0) 
    mortality_df.to_csv(Data_save+'mortality.csv')
    
    age_df=pd.concat(ages,0) 
    age_df.to_csv(Data_save+'age.csv')
    
    los_df=pd.concat(los,0) 
    los_df.to_csv(Data_save+'los.csv')
    
    sepsis_hour_df=pd.concat(sepsis_hours,0) 
    sepsis_hour_df.to_csv(Data_save+'sepsis_hour.csv')