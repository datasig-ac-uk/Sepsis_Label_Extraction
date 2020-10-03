from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score,roc_curve
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from src.features.sepsis_mimic3_myfunction import *

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
    prob_preds=np.empty((0,1),int)
    
    k=len(tra_full_indices)
    
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



