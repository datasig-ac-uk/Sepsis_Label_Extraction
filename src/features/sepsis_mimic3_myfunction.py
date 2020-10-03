import numpy as np
import pandas as pd
import iisignature
import os
import random

import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.model_selection import KFold

from dicts import *


def create_folder(path):
    try:
        # Create target directory
        os.makedirs(path)
        print("Directory " , path ,  " created ") 
    except FileExistsError:
        print("Directory " , path,  " already exists")   

################################### Feature extractions     ######################################## 

def compute_icu(df_merge2,definition,return_results=False):
    
    nonsepsis_icu_number=len(sorted(list(df_merge2[(df_merge2[definition].isnull())].icustay_id.unique())))
    
    
    icu_number=len(sorted(list(df_merge2.icustay_id.unique())))
    print('ICUStay id number:',icu_number)
    
    print('Sepsis ICUStay number and Sepsis ICU ratio:',icu_number-nonsepsis_icu_number,\
          (icu_number-nonsepsis_icu_number)/icu_number)
    
    if return_results:
        return icu_number,icu_number-nonsepsis_icu_number, (icu_number-nonsepsis_icu_number)/icu_number
    
    
def partial_sofa(df):
    """ Partial reconstruction of the SOFA score from features available in the sepsis dataset. """
    # Init the tensor
    
    sofa = np.full([len(df)], np.nan)
    platelets=np.full([len(df)], np.nan)
    bilirubin_total=np.full([len(df)], np.nan)
    maps=np.full([len(df)], np.nan)
    creatinine=np.full([len(df)], np.nan)
   

    # Coagulation
    platelets_ =  df['heart_rate']
    
    platelets[platelets_ >= 150] = 0
    platelets[(100 <= platelets_) & (platelets_ < 150)] = 1
    platelets[(50 <= platelets_) & (platelets_ < 100)] = 2
    platelets[(20 <= platelets_) & (platelets_ < 50)] = 3
    platelets[platelets < 20] = 4

    # Liver
    bilirubin_ = df['bilirubin_total']
    
    bilirubin_total[bilirubin_ < 1.2] = 0
    bilirubin_total[(1.2 <= bilirubin_) & (bilirubin_ <= 1.9)] = 1
    bilirubin_total[(1.9 < bilirubin_) & (bilirubin_ <= 5.9)] = 2
    bilirubin_total[(5.9 < bilirubin_) & (bilirubin_ <= 11.9)] = 3
    bilirubin_total[bilirubin_ > 11.9] = 4

    # Cardiovascular
    map_ = df['mean_airway_pressure']
    maps[map_ >= 70] = 0
    maps[map_ < 70] = 1

    # Creatinine
    creatinine_ = df['creatinine']
    creatinine[creatinine_ < 1.2] = 0
    creatinine[(1.2 <= creatinine_) & (creatinine_ <= 1.9)] = 1
    creatinine[(1.9 < creatinine_) & (creatinine_ <= 3.4)] = 2
    creatinine[(3.4 < creatinine_) & (creatinine_ <= 4.9)] = 3
    creatinine[creatinine_ > 4.9] = 4
    
    
    sofa=platelets+bilirubin_total+maps+creatinine
    
    return sofa

def dataframe_from_definition_discard(path_df, a1=6,a2=3,definition='t_sepsis_min'):
    """
        For specific definition among the three, deleting patients who has been diagnosied
        beofre a1 hours in icu and trimming valid patients' data such that all the data after
        a2 hours after sepsis onset according to definition will be deleted. 
        
        
    nonsepsis==True: we keep nonsepsis data; otherwsie, we do not keep them
        
    """
    
    pd.set_option('display.max_columns', None)


    ##Reading df:


    df_sepsis = pd.read_csv(path_df,\
                        low_memory=False,
                        parse_dates=['admittime', 'dischtime', 'deathtime', 'intime', 'outtime',
                              'charttime','t_suspicion', 't_sofa', 't_sepsis_min', 't_sepsis_max'])

    df_sepsis['floored_charttime'] = df_sepsis['charttime'].dt.floor('h')
    
    df_sepsis=df_sepsis.sort_values(['icustay_id','floored_charttime'])
    print('Initial size:',df_sepsis.deathtime.size)
    
    ### deleting rows that patients are dead already 
    df_sepsis=df_sepsis[(df_sepsis['deathtime']>df_sepsis['charttime']) |df_sepsis.deathtime.isnull()]  
    df_sepsis['gender'] = df_sepsis['gender'].replace(('M', 'F'), (1, 0))

    ### Newly added by Lingyi to fill in the gaps
    df_sepsis_intime = df_sepsis[identifier + static_vars].sort_values(['icustay_id', 'floored_charttime']).groupby(['icustay_id'], as_index = False).first()
    df_sepsis_intime['floored_intime'] = df_sepsis_intime['intime'].dt.floor('h')
    df_sepsis_intime2 = df_sepsis_intime[df_sepsis_intime['floored_charttime'] > df_sepsis_intime['floored_intime']]
    df_sepsis_intime2.drop(['floored_charttime'], axis = 1, inplace=True)
    df_sepsis_intime2.rename(columns={"floored_intime": "floored_charttime"}, inplace=True)
    df_sepsis = pd.concat([df_sepsis, df_sepsis_intime2], ignore_index=True, sort=False).sort_values(['icustay_id', 'floored_charttime'])

    
    ### Discarding patients developing sepsis within 4 hour in ICU
    df_sepsis=df_sepsis[(df_sepsis[definition].between(df_sepsis.intime+pd.Timedelta(hours=4),\
                                                            df_sepsis.outtime, inclusive=True))\
                             | (df_sepsis[definition].isnull())]
    
    print("Size of instances after discarding patients developing sepsis within 4 hour in ICU:",df_sepsis.deathtime.size)
    
    df_first = df_sepsis[static_vars+categorical_vars+identifier].groupby(by=identifier, as_index = False).first()
    df_mean = df_sepsis[numerical_vars+identifier].groupby(by=identifier, as_index = False).mean()
    df_max = df_sepsis[flags+identifier].groupby(by=identifier, as_index = False).max()
    df_merge = df_first.merge(df_mean, on= identifier).merge(df_max, on= identifier)
    df_merge.equals(df_merge.sort_values(identifier))    
    df_merge.set_index('icustay_id',inplace=True)
    
    ### Resampling
    df_merge2 = df_merge.groupby(by='icustay_id').apply(lambda x : x.set_index('floored_charttime').\
                                                       resample('H').first()).reset_index()
    
    print("Size after averaging hourly measurement and resampling:",df_merge2.deathtime.size)
    
    df_merge2[['subject_id', 'hadm_id', 'icustay_id']+static_vars] = df_merge2[['subject_id', 'hadm_id', 'icustay_id']+static_vars].groupby('icustay_id', as_index=False).apply(lambda v: v.ffill())
    
    ### Deleting cencored data after a2
    df_merge2=df_merge2[((df_merge2.floored_charttime-df_merge2[definition]).dt.total_seconds()/60/60<a2+1.0)\
                        |(df_merge2[definition].isnull())]
    
    print("Size of instances after getting censored data:",df_merge2.deathtime.size)
    
    ### Adding icu stay since intime
    df_merge2['rolling_los_icu'] = (df_merge2['outtime'] - df_merge2['intime']).dt.total_seconds()/60/60
    df_merge2['hour'] = (df_merge2['floored_charttime'] - df_merge2['intime']).dt.total_seconds()/60/60
    
    ### Discarding patients staying less than 4 hour or longer than 20 days
    df_merge2=df_merge2[(df_merge2['rolling_los_icu']<=20*24) & (4.0<=df_merge2['rolling_los_icu'])]
    
    print("Size of instances after discarding patients staying less than 4 hour or longer than 20 days:",df_merge2.deathtime.size)
    
    ### Triming the data to icu instances only
    df_merge2 = df_merge2[df_merge2['floored_charttime'] <= df_merge2['outtime']]
    df_merge2 = df_merge2[df_merge2['intime'] <= df_merge2['floored_charttime']]
    print("After triming the data to icu instances:",df_merge2.deathtime.size)
    
    icustay_id_to_included_bool=(df_merge2.groupby(['icustay_id']).size()>=4)
    icustay_id_to_included=df_merge2.icustay_id.unique()[icustay_id_to_included_bool]
    
    df_merge2=df_merge2[df_merge2.icustay_id.isin(icustay_id_to_included)]
    
    ### Mortality check
    df_merge2['mortality']=(~df_merge2.deathtime.isnull()).astype(int)
    df_merge2['sepsis_hour'] = (df_merge2[definition] - df_merge2['intime']).dt.total_seconds()/60/60
    
    print("Final size:",df_merge2.deathtime.size)

    return df_merge2

def jamesfeature(df,Data_Dir='./',definition='t_sepsis_min',save=True):
        
        

    
        df['HospAdmTime']=(df['admittime'] - df['intime']).dt.total_seconds()/60/60
        
        
        print('Finally getting james feature:')
        james_feature=lgbm_jm_transform(df)
        
        if save:
            np.save(Data_Dir+'james_features'+definition[1:]+'.npy', james_feature)

        print('Size of james feature for definition', james_feature.shape)

        return james_feature  

    
def get_list(x,m):
    if len(x)>=6:
        return [[np.nan for j in range(m)] for i in range(m-1)]+list(zip(*(x[i:] for i in range(m))))
    else:
        return [[np.nan for j in range(m)] for i in range(len(x))]
    
    

def LL(datapiece):
    
    a=np.repeat(datapiece.reshape(-1,1), 2, axis=0)
    return np.concatenate((a[1:,],a[:-1,:]),axis=1)


    
def signature(datapiece,s=iisignature.prepare(2,3),\
              order=3, leadlag=True, logsig=True):

    if leadlag:
        if logsig:
            return np.asarray([iisignature.logsig(LL(datapiece[i,:]), s) for i in range(datapiece.shape[0])])
        else:
                
            return np.asarray([iisignature.sig(LL(datapiece[i,:]), order) for i in range(datapiece.shape[0])])
    else:

                
            return np.asarray([iisignature.sig(datapiece[i,:], order) for i in range(datapiece.shape[0])])





def lgbm_jm_transform(dataframe1,feature_dict=feature_dict,\
                      windows1=8,windows2=7,windows3=6,\
                      order=3,leadlag=True,logsig=True):
    
    
    """
        Directly replicating how James extracted features.
    """
    
    count_variables = feature_dict_james['laboratory'] + ['temp_celcius']
    
    rolling_count=dataframe1.groupby('icustay_id').apply(lambda x: x[count_variables].rolling(windows1).count())
    newdata1=np.asarray(rolling_count)
    
    ## current number 27
    
    dataframe1 = dataframe1.groupby('icustay_id', as_index=False).apply(lambda v: v.ffill())
        
    dataframe1['bun_creatinine']=dataframe1['bun']/dataframe1['creatinine']
    dataframe1['partial_sofa']=partial_sofa(dataframe1)
    dataframe1['shock_index']=dataframe1['heart_rate']/dataframe1['nbp_sys']
    
    newdata=[]
    
    for variable in feature_dict_james['vitals']:
        
        
        temp_var=np.array(dataframe1.groupby('icustay_id').apply(lambda x: x[variable].rolling(windows2).var()))
        temp_mean=np.array(dataframe1.groupby('icustay_id').apply(lambda x: x[variable].rolling(windows2).mean()))
        
        newdata.append(temp_var+temp_mean**2)
        
        temp_max=np.array(dataframe1.groupby('icustay_id').apply(lambda x: x[variable].rolling(windows3).max()))
        temp_min=np.array(dataframe1.groupby('icustay_id').apply(lambda x: x[variable].rolling(windows3).min()))
        
        newdata.append(temp_max)
        newdata.append(temp_min)
    
    newdata=np.asarray(newdata).T
    ## current number 3*7=21
    ##total 21+27=48
    
    newdata=np.concatenate((newdata1, newdata),axis=1)
    
    sig_features=['heart_rate','nbp_sys', 'abp_mean']+feature_dict_james['derived'][:-1]
    
    for j in range(len(sig_features)):   

            variable=sig_features[j]
 
            if leadlag:
                if logsig:
                    siglength=iisignature.logsiglength(2,order)
                else:
                    siglength=iisignature.siglength(2,order) 
            else:
                siglength=iisignature.siglength(1,order)
                
            sigfeature=np.empty((0,siglength),float)
                        
            ## To extract all rolling windows of each icustay_id            
            rollingwindows=dataframe1.groupby('icustay_id').apply(lambda x:get_list(x[variable],windows2))
            
            ## apply signature for all rolling windows of each icustay_id at once
            for jj in range(len(rollingwindows)):
                
                rolling_data=signature(np.asarray(rollingwindows.values[jj]),order=order,leadlag=leadlag,logsig=logsig)
                sigfeature=np.concatenate((sigfeature, rolling_data),axis=0) 
                
            assert newdata.shape[0]==sigfeature.shape[0]

            newdata=np.concatenate((newdata, sigfeature),axis=1)
    
    ## current step number 5*5=25
    ##total so far 48+25=73
    
    #Currently, we finished adding new features, need to concat with original data
    ##concat original dataset

    original_features=dataframe1[feature_dict_james['vitals']+feature_dict_james['laboratory']\
                                 +feature_dict_james['derived']+['age','gender','hour','HospAdmTime']]
    original_features=np.asarray(original_features)
    
    ## current step number 7+26+3+4=40
    ##total so far 73+40=113
    
    finaldata=np.concatenate((original_features,newdata),axis=1)

    return finaldata


def dataset_generator(icustay_lengths,features):
        
        """
        Generating dataset for lstm from features
        
        """

        index = np.cumsum(np.array([0] + icustay_lengths))
        features_list = [torch.tensor(features[index[i]:index[i + 1]]) for i in range(index.shape[0] - 1)]
        column_list = [item for item in range(features.shape[1])]
        dataset = TimeSeriesDataset(data=features_list, columns=column_list, lengths=icustay_lengths)
        dataset.data = torch_ffill(dataset.data)
        dataset.data[torch.isnan(dataset.data)] = 0
        dataset.data[torch.isinf(dataset.data)] = 0
        
        return dataset

def data_generator(path_df,Save_Dir,a2=0, T_list=[12,8,12,4],\
                   definitions=['t_sofa','t_suspicion','t_sepsis_min']):
    
    create_folder(Save_Dir)
    
    results=[]
    
    for definition in definitions:
        
        a1=6
        
        print('definition = '+str(definition))
        
        print('generate features on data set')       
        df_sepsis1 = dataframe_from_definition_discard(path_df, definition=definition,a1=a1,a2=a2)
        
        print('save septic ratio for data set')      
        icu_number,sepsis_icu_number, septic_ratio=compute_icu(df_sepsis1,definition,return_results=True)
        results.append([str(x)+','+str(y),definition,icu_number,sepsis_icu_number, septic_ratio])

        print('save ICU Ids for data set')       
        icuid_sequence=df_sepsis1.icustay_id.unique()
        np.save(Save_Dir +'icustay_id'+definition[1:]+'.npy',icuid_sequence)
        
        print('save ICU lengths for data set')     
        icustay_lengths=list(df_sepsis1_train.groupby('icustay_id').size())
        np.save(Save_Dir +'icustay_lengths'+definition[1:]+'.npy',icustay_lengths)

        print('save processed dataframe for lstm model')
        df_sepsis1.to_pickle(Save_Dir+definition[1:]+'_dataframe.pkl')
        
        print('generate and save input features')
        features = jamesfeature(df_sepsis1, Data_Dir=Save_Dir, definition=definition)
 
        print('generate and save timeseries dataset for LSTM model input')    
        dataset=dataset_generator(icustay_lengths,features)
        
        dataset.save(Save_Dir + definition[1:] + '_ffill.tsd')
            
        print('gengerate and save labels')
        for T in T_list:
            print('T= ' + str(T))
            labels = label_generator(df_sepsis1, a1=T, Data_Dir=Save_Dir, definition=definition, save=True)
            
    print('save icu spetic ratio to csv')               
    result_df = pd.DataFrame(results, columns=['x,y', 'definition', 'total_icu_no','sepsis_no','septic_ratio'])
    result_df.to_csv(Save_Dir+'icu_number.csv')     



################################### CV splitting  ########################################   



def cv_bundle(full_indices,shuffled_indices,k=5):
    
    """
    conduct cv splitting given full indices
    
    """
    
    tra_patient_indices,tra_full_indices,val_patient_indices,val_full_indices=[],[],[],[]
    
    kf = KFold(n_splits=k)
    for a, b in kf.split(shuffled_indices):

        tra_patient_indices.append(shuffled_indices[a])
        val_patient_indices.append(shuffled_indices[b])
        
        tra_full_indices.append([full_indices[tra_patient_indices[-1][i]] for i in range(len(a))])
        val_full_indices.append([full_indices[val_patient_indices[-1][i]] for i in range(len(b))])
    
    return tra_patient_indices,tra_full_indices,val_patient_indices,val_full_indices

def patient_idx(icustay_lengths):
    
    """
        idxs for each patient, [[idx for patient 1],..[idx for patient i],...]
    
    """
    
    icustay_lengths_cumsum=np.insert(np.cumsum(np.array(icustay_lengths)),0,0)
    total_indices=len(icustay_lengths)
    icustay_fullindices=[np.arange(icustay_lengths[i])+icustay_lengths_cumsum[i] for i in range(total_indices)] 
       
    return icustay_fullindices 

def dataframe_cv_pack(df_sepsis,k=5,definition='t_sepsis_min',path_save='./',save=True):
    """
    
        Outputing the data lengths for each patient and outputing k cv splitting.
        
        
        We store/call the random indice with name
        
            path_save+'icustay_lengths'+definition[1:]+'.npy'
            path_save+'shuffled_indices'+definition[1:]+'.npy'
            
    """
    if not save:
        
        icustay_lengths=np.load(path_save+'icustay_lengths'+definition[1:]+'.npy')  
        shuffled_indices=np.load(path_save+'shuffled_indices'+definition[1:]+'.npy')
        total_indices=len(shuffled_indices)

    else:
        icustay_id=sorted(list(df_sepsis.icustay_id.unique()))
        total_indices=len(icustay_id)
    
        icustay_lengths=list(df_sepsis.groupby('icustay_id').size())
        np.save(path_save+'icustay_lengths'+definition[1:]+'.npy',icustay_lengths)

        
        shuffled_indices=np.arange(total_indices)
        random.seed(42)
        random.shuffle(shuffled_indices)
        np.save(path_save+'shuffled_indices'+definition[1:]+'.npy',shuffled_indices)
        
    #Getting the full indices      
    icustay_fullindices=patient_idx(icustay_lengths)
    
    tra_patient_indices,tra_full_indices,val_patient_indices,val_full_indices=cv_bundle(icustay_fullindices,\
                                                                                        shuffled_indices,\
                                                                                         k=k)

    return icustay_lengths, tra_patient_indices,tra_full_indices,val_patient_indices,val_full_indices

def cv_pack(icustay_lengths,k=5,definition='t_sepsis_min',path_save='./',save=True):
    
    """
    
        outputing k cv splitting.
        
        
        We store/call the random indice with name
        
            path_save+'icustay_lengths'+definition[1:]+'.npy'
            path_save+'shuffled_indices'+definition[1:]+'.npy'
            
    """
    if not save:
        
        shuffled_indices=np.load(path_save+'shuffled_indices'+definition[1:]+'.npy')
        total_indices=len(shuffled_indices)

    else:
        total_indices=len(icustay_lengths)        
        shuffled_indices=np.arange(total_indices)
        random.seed(42)
        random.shuffle(shuffled_indices)
        np.save(path_save+'shuffled_indices'+definition[1:]+'.npy',shuffled_indices)
        
    #Getting the full indices      
    icustay_fullindices=patient_idx(icustay_lengths)
    
    tra_patient_indices,tra_full_indices,val_patient_indices,val_full_indices=cv_bundle(icustay_fullindices,\
                                                                                        shuffled_indices,\
                                                                                         k=k)

    return tra_patient_indices,tra_full_indices,val_patient_indices,val_full_indices


################################### For train/test set  without cv splitting ########################################   



def icu_lengths(df_sepsis,definition='t_sepsis_min',path_save='./',save=True):
    """
    
        Outputing/save the data lengths for each patient.

            path_save+'icustay_lengths'+definition[1:]+'.npy';
            
    """
    if not save:
        
        icustay_lengths=np.load(path_save+'icustay_lengths'+definition[1:]+'.npy')  

    else:
        icustay_id=sorted(list(df_sepsis.icustay_id.unique()))
        total_indices=len(icustay_id)
        icustay_lengths=list(df_sepsis.groupby('icustay_id').size())
        np.save(path_save+'icustay_lengths'+definition[1:]+'.npy',icustay_lengths)

    return icustay_lengths

################################### Outputing labels ########################################   

def labels_generator(df, a1=6,Data_Dir='./',definition='t_sepsis_min',save=True):
    
        """
            Generating labels for dataset df in dataframe.
            
        """
    
        print('Labeling:')
        a=list((df[definition]-df.floored_charttime).dt.total_seconds()/60/60<=a1)   
        label=np.array([int(a[i]) for i in range(len(a))])
        if save:
            np.save(Data_Dir+'label'+definition[1:]+'_'+str(a1)+'.npy', label)
        print('length of label',len(label))

        return label

def labels_validation(labels, val_full_indices):
    """
        labels on validation sets
    
    """
    
    labels_true=np.empty((0,1),int)   
    k=len(test_full_indices)
    
    for i in range(k):

        val_binary_labels = labels[np.concatenate(val_full_indices[i])]        
        labels_true=np.append(labels_true, val_binary_labels)    

       
    return labels_true
