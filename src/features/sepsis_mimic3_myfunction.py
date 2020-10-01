import numpy as np
import pandas as pd
import iisignature
import os
import random

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score,roc_curve
from sklearn import preprocessing
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression

from dicts import *
from optimizers import *

def create_folder(path):
    try:
        # Create target directory
        os.mkdir(path)
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
        
        df['SepsisLabel']=label
        
        return label, scores.values


################################### LGBM tuning/training ########################################   

    
def model_validation(model, dataset, labels, tra_full_indices, val_full_indices):
    
    """
        
        For chosen model, and for dataset after all necessary transforms, we conduct k-fold cross-validation simutaneously.

        Input:
            model
            dataset: numpy version
            labels: numpy array
            tra_full_indices/val_full_indices: from cv splitting
            
        Output:
            tra_preds:  predicted labels on concatenated tra sets
            prob_preds: predicted risk scores for the predicted labels
            labels_true: true labels for tra_preds
            auc score
            sepcificity at fixed sensitivity level
            accuracy at fixed sensitivity level
        
    """    

    labels_true=np.empty((0,1),int)   
    tra_preds=np.empty((0,1),int)
    tra_idxs=np.empty((0,1),int)
    k=len(train_full_indices)

    val_idx_collection=[]
    
    for i in range(k):
        
        print('Now training on the', i, 'splitting')
        
        tra_dataset=dataset[np.concatenate(tra_full_indices[i]),:]        
        val_dataset =dataset[np.concatenate(val_full_indices[i]),:]
        
        tra_binary_labels = labels[np.concatenate(tra_full_indices[i])]
        val_binary_labels = labels[np.concatenate(val_full_indices[i])]
        
        model.fit(X=tra_dataset,y=tra_binary_labels)  
            
        predicted_prob=model.predict_proba(val_dataset)[:,1]
        prob_preds=np.append(prob_preds,predicted_prob)  
        tra_idxs=np.append(tra_idxs,np.concatenate(val_full_indices[i]))
        labels_true=np.append(labels_true, val_binary_labels)    


    fpr, tpr, thresholds = roc_curve(labels_true, prob_preds, pos_label=1)
    index=np.where(tpr>=0.85)[0][0]

    tra_preds=np.append(tra_preds,(prob_preds>=thresholds[index]).astype('int'))        
    print('auc and sepcificity',roc_auc_score(labels_true,prob_preds),1-fpr[index])
    print('accuracy',accuracy_score(labels_true,tra_preds))
        
    return tra_preds, prob_preds, labels_true,\
           roc_auc_score(labels_true,prob_preds),\
           1-fpr[index],accuracy_score(labels_true,tra_preds)


grid_parameters ={ # LightGBM
        'n_estimators': [40,70,100,200,400,500, 800],
        'learning_rate': [0.08,0.1,0.12,0.05],
        'colsample_bytree': [0.5,0.6,0.7, 0.8],
        'max_depth': [4,5,6,7,8],
        'num_leaves': [5,10,16, 20,25,36,49],
        'reg_alpha': [0.001,0.01,0.05,0.1,0.5,1,2,5,10,20,50,100],
        'reg_lambda': [0.001,0.01,0.05,0.1,0.5,1,2,5,10,20,50,100],
        'min_split_gain': [0.0,0.1,0.2,0.3, 0.4],
        'subsample': np.arange(10)[5:]/12,
        'subsample_freq': [10, 20],
        'max_bin': [100, 250,500,1000],
        'min_child_samples': [49,99,159,199,259,299],
        'min_child_weight': np.arange(30)+20}

def model_tuning(model, dataset, labels,tra_full_indices, val_full_indices,param_grid,\
                      grid=False,n_iter=100, n_jobs=-1, scoring='roc_auc',verbose=2):
    
    """
        
        For chosen base model, we conduct hyperparameter-tuning on given cv splitting.
        
        Input:
            model
            dataset: numpy version
            labels: numpy array
            tra_full_indices/val_full_indices: from cv splitting
            param_grid:for hyperparameter tuning
        
        Output:
            
            set of best parameters for base model
        
        
    """    


    k=len(tra_full_indices)
    
    cv=[[np.concatenate(tra_full_indices[i]),np.concatenate(val_full_indices[i])] for i in range(k)]
    
    if grid:
                gs = GridSearchCV(estimator=model, \
                                  param_grid=param_grid,\
                                  n_jobs=n_jobs,\
                                  cv=cv,\
                                  scoring=scoring,\
                                  verbose=verbose)
    else:
        
                gs = RandomizedSearchCV(model, \
                                        param_grid,\
                                        n_jobs=n_jobs,\
                                        n_iter=n_iter,\
                                        cv=cv,\
                                        scoring=scoring,\
                                        verbose=verbose)  
        
    fitted_model=gs.fit(X=dataset,y=labels)
    best_params_=fitted_model.best_params_
    
    return best_params_        
       



def model_training(model, train_set,test_set, train_labels, test_labels):
    
        """
        
        For chosen model, conduct standard training and testing
        
        Input:
            model
            train_set,test_set: numpy version
            train_labels, test_labels: numpy array
             
        Output:
            test_preds:  predicted labels on test set (numpy array)
            prob_preds_test: predicted risk scores for the predicted test labels (numpy array)
            
            auc score
            sepcificity at fixed sensitivity level
            accuracy at fixed sensitivity level

        
        
        """        
    
        model.fit(X=train_set,y=train_labels)              
        prob_preds_test=model.predict_proba(test_set)[:,1]


        print('Test:')
        fpr, tpr, thresholds = roc_curve(test_labels, prob_preds_test, pos_label=1)
        index=np.where(tpr>=0.85)[0][0]
        test_preds=np.array((prob_preds_test>=thresholds[index]).astype('int'))

        print('auc and sepcificity',roc_auc_score(labels2,prob_preds_test),1-fpr[index])
        print('accuracy',accuracy_score(test_labels,test_preds))
        
        return test_preds, prob_preds_test, roc_auc_score(labels2,prob_preds_test),\
               1-fpr[index],accuracy_score(labels2,test_preds)





def labels_validation(labels, test_full_indices):
    """
        labels on validation sets
    
    """
    
    labels_true=np.empty((0,1),int)   
    k=len(test_full_indices)
    
    for i in range(k):

        val_binary_labels = labels[np.concatenate(test_full_indices[i])]        
        labels_true=np.append(labels_true, val_binary_labels)    

       
    return labels_true



