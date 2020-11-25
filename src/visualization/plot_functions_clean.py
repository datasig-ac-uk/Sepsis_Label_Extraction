import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve, auc

from matplotlib.patches import Rectangle

    
import sys
sys.path.insert(0, '../../')

from definitions import *

colors_barplot=sns.color_palette()
colors_auc=sns.color_palette("Dark2")
linestyles=[':','-.','-','--']        




############################ For auc plots ############################

def auc_plot(trues_list,probs_list,names,fontsize=14,\
             colors=colors_auc,linestyles=linestyles,\
             lw = 2,loc="lower right", save_name=None):
    
    """
        AUC plots in one figure via ground truth and predicted probabilities
        
    Input:
    
        trues_list: ground-truth-seq list 
        
                eg, for 2 set of data, [[ground truth for set1],[ground truth for set2]]
                
        probs_list: probability-seq list
        
            eg, for 2 set of data, [[probabilities for set1],[probabilities for set2]]
            
        names: curve labels
        
        save_name: if None: print figure; else: save to save_name.png
        
    """
    
    num=len(trues_list)
    
    plt.figure()
    
    for i in range(num):
        
        fpr, tpr, _ = roc_curve(trues_list[i], probs_list[i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.ylabel('True Positive Rate',fontsize=fontsize)
    plt.legend(loc=loc,fontsize=fontsize-3)
    
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    if save_name is not None:
        plt.savefig(save_name+'.jpeg',dpi=350)
    else:        
        plt.show()

        
#########################For CI ################################################
from scipy.stats import norm

n_bootstraps = 100
#rng_seed = 1  # control reproducibility
alpha = 0.95

def CI_AUC_bootstrapping(n_bootstraps, alpha, y_true, y_pred, rng_seed = 1):
    # to compute alpha % confidence interval using boostraps for n_boostraps times 
    bootstrapped_scores = []
    fprs,tprs=[],[]
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        #sample_index = np.random.choice(range(0, len(y_pred)), len(y_pred))
       # print(indices)

        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
                continue
            

        score = roc_auc_score(y_true[indices], y_pred[indices])
        fpr, tpr, _= roc_curve(y_true[indices],y_pred[indices])
        fprs.append(fpr)
        tprs.append(tpr)
        bootstrapped_scores.append(score)
#         if i%20 ==0:
#             print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
            
    factor =  norm.ppf(alpha)
    std1 = np.std(bootstrapped_scores)
    mean1 = np.mean(bootstrapped_scores)
    up1 = mean1+factor*std1
    lower1 =  mean1-factor*std1
#     print( '{}% confidence interval is [{},{}]'.format(alpha, up1, lower1))
    return [lower1,up1],fprs,tprs

def fprs_tprs_output(labels_list_list,probs_list_list,n_bootstraps=100,alpha=0.95):

    fprs_lists=[[] for kk in range(len(MODELS))]
    tprs_lists=[[] for kk in range(len(MODELS))]


    for i in range(len(MODELS)):
    
        fprs_lists[i]=[[] for k in range(len(definitions))]
        tprs_lists[i]=[[] for k in range(len(definitions))]   
    
        for j in range(len(definitions)):
        
            CI_results,fprs,tprs=CI_AUC_bootstrapping(n_bootstraps, alpha,labels_list_list[i][j],\
                                                              probs_list_list[i][j],  rng_seed = 1)
        
            print(MODELS[i],definitions[j],"{:.3f}".format(roc_auc_score(labels_list_list[i][j],\
                                                                         probs_list_list[i][j])),\
              "["+"{:.3f}".format(CI_results[0]) +","+"{:.3f}".format(CI_results[1])+"]")

        
            fprs_lists[i][j]+=fprs
            tprs_lists[i][j]+=tprs

        print('\n')
        
    return fprs_lists, tprs_lists

def CI_std_output(fprs_lists,tprs_lists,\
                  mean_fpr_list=[np.linspace(0, 1, 30+0*i) for i in range(3)]):



    error_list=[[] for i in range(len(MODELS))]

    for i in range(len(MODELS)):

        error_list[i]=[[] for k in range(len(definitions))]
    
    
        for j in range(len(definitions)):
            tprs_=[]
        
            for k in range(len(tprs_lists[i][j])):
            
                fpr_now=fprs_lists[i][j][k]
                tpr_now=tprs_lists[i][j][k]
                interp_tpr = np.interp(mean_fpr_list[i], fpr_now, tpr_now)
                interp_tpr[0] = 0.0
                tprs_.append(interp_tpr)
        
            mean_tpr = np.mean(tprs_, axis=0)
            mean_tpr[-1] = 1.0
            
            std_tpr = np.std(tprs_, axis=0)
            error_list[i][j]=std_tpr
            
    return error_list


# colors_shade=sns.color_palette("Set2")

colors_shade=sns.color_palette("Pastel2")
def auc_subplots(trues_list,probs_list,names,tprs_lists=None,fontsize=14,figsize=(15,5),\
                 titles=MODELS, colors=colors_auc, colors_shade=colors_shade,\
                 linestyles=linestyles,lw=2, loc="lower right",save_name=None):
    
    """
        
        AUC plots for different models via ground truth and predicted probabilities
        
    Input:
    
        trues_list: list of ground-truth-seq lists 
        
                eg, for three models for 2 set of data,[[model1 truth-list], [model2 truth-list], [model3 truth-list]]
                    [model1 truth-list]=[[ground truth for set1],[ground truth for set2]]
                
        probs_list: probability-seq list
        
            eg, for three models for 2 set of data,  [[model1 probs-list], [model2 probs-list], [model3 probs-list]]
            
                [model1 probs-list]=[[probabilities for set1],[probabilities for set2]]
            
        names: curve labels for sets of data
        
        save_name: if None: print figure; else: save to save_name.png

        
    
    """

    plt.figure(figsize=figsize)
    plt.subplot(131)
    
    num=len(trues_list[0])
    
    for i in range(num):
        
        fpr, tpr, _ = roc_curve(trues_list[0][i], probs_list[0][i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        
        if tprs_lists is not None:
            plt.fill_between(tprs_lists[0], tprs_lists[1][0][i], tprs_lists[-1][0][i], color=colors_shade[i], alpha=.2)
        
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.ylabel('True Positive Rate',fontsize=fontsize)
    plt.title(titles[0])
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    plt.subplot(132)
    
    num=len(trues_list[1])
    for i in range(num):
        
        fpr, tpr, _ = roc_curve(trues_list[1][i], probs_list[1][i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)

        if tprs_lists is not None:
            plt.fill_between(tprs_lists[0], tprs_lists[1][1][i], tprs_lists[-1][1][i], color=colors_shade[i], alpha=.2)

        
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.title(titles[1])
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    
    plt.subplot(133)
    
    num=len(trues_list[-1])
    for i in range(num):
        
        fpr, tpr, _ = roc_curve(trues_list[-1][i], probs_list[-1][i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)

        if tprs_lists is not None:
            plt.fill_between(tprs_lists[0], tprs_lists[1][-1][i], tprs_lists[-1][-1][i], color=colors_shade[i], alpha=.2)
      
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.title(titles[-1])
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    if save_name is not None:
        plt.savefig(save_name+'.jpeg',dpi=350)
    else:
        
        plt.show()

        
        ############## Patient level ####################
        
def patient_level_probability_max(probs):
    
    return np.max(probs)

def fprs_tprs_output_patient_level(labels_list_list,probs_list_list,indices_list_list,n_bootstraps=100,alpha=0.95):

    fprs_lists=[[] for kk in range(len(MODELS))]
    tprs_lists=[[] for kk in range(len(MODELS))]

    
    probs_list=[[] for kk in range(len(MODELS))]
    labels_list=[[] for kk in range(len(MODELS))]
    
    for i in range(len(MODELS)):
    
        fprs_lists[i]=[[] for k in range(len(definitions))]
        tprs_lists[i]=[[] for k in range(len(definitions))]   
        
        probs_list[i]=[[] for k in range(len(definitions))]
        labels_list[i]=[[] for k in range(len(definitions))]   
        
        for j in range(len(definitions)):

            full_idxs=indices_list_list[i][j]
            par_probs=[patient_level_probability_max(probs_list_list[i][j][full_idxs[k]]) for k in range(len(full_idxs))]

            par_labels=[labels_list_list[i][j][full_idxs[k]][-1] for k in range(len(full_idxs))]
        

            CI_results,fprs,tprs=CI_AUC_bootstrapping(n_bootstraps, alpha, np.array(par_labels), np.array(par_probs),  rng_seed = 1)
        
            print(MODELS[i],definitions[j],\
                  "{:.3f}".format(roc_auc_score(par_labels,par_probs)),\
                  "["+"{:.3f}".format(CI_results[0]) +","+"{:.3f}".format(CI_results[1])+"]")

        
            fprs_lists[i][j]+=fprs
            tprs_lists[i][j]+=tprs
            
            probs_list[i][j]=np.array(par_probs)
            labels_list[i][j]=np.array(par_labels)

        print('\n')
        
    return fprs_lists, tprs_lists,labels_list,probs_list

def recall_specificity_subplots_patient_level(pres_list,tprs_list,names,\
                                              fontsize=14,figsize=(15,5),\
                                              titles=MODELS, colors=colors_auc,\
                                              linestyles=linestyles,\
                                              loc="lower left",lw = 2,\
                                              save_name=None):
               
    """
        
        
        
        recall_specificity plots for different models via computed precisions and tprs
        
    Input:
    
        pres_list: list of precision lists 
        
                eg, for three models for 2 set of data,[[model1 precision-list], [model2 precision-list], [model3 precision-list]]
                    [model1 precision-list]=[[precision for data set1],[precision for data set2]]
                
        tprs_list: list of tpr list
        
                eg, for three models for 2 set of data,[[model1 tpr-list], [model2 tpr-list], [model3 tpr-list]]
                    [model1 tpr-list]=[[tpr for data set1],[tpr for data set2]]

            
        names: curve labels for sets of data
        
        save_name: if None: print figure; else: save to save_name.png

    
    
    """

    plt.figure(figsize=figsize)
    plt.subplot(131)
    
    
    num=len(tprs_list[0])    

    for i in range(num):

        plt.plot(pres_list[0][i], tprs_list[0][i], color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label=names[i])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision',fontsize=fontsize)
    plt.ylabel('Recall',fontsize=fontsize)
    plt.title(titles[0])
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    plt.subplot(132)
    
    num=len(tprs_list[1])    

    for i in range(num):
        
        plt.plot(pres_list[1][i], tprs_list[1][i], color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label=names[i])

    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision',fontsize=fontsize)
    plt.title(titles[1])
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)    
    
    plt.subplot(133)

    num=len(tprs_list[-1])    

    for i in range(num):

        plt.plot(pres_list[0][i], tprs_list[0][i], color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label=names[i])

    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision',fontsize=fontsize)
    plt.title(titles[-1])
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    if save_name is not None:
        plt.savefig(save_name+'.jpeg',dpi=350)
    else:
        
        plt.show()

        

############################ For trajectory level plot ############################
               
def finding_icuid_idx(idx1_septic,test_patient_indices,icustay_lengths=None,\
                      Data_Dir='./Sep_24_12_experiments_new/icustay_id',\
                      definition='t_suspicion'):
    
    idx1_septic_original=np.concatenate(test_patient_indices)[idx1_septic]        
    
    if icustay_lengths is not None:
        print('The length of current patient',icustay_lengths[idx1_septic_original])
        
    icuid_sequence=np.load(Data_Dir+definition[1:]+'.npy')
        
    return icuid_sequence[idx1_septic_original]
    

def finding_sample_idx(idx1_septicicuid,test_patient_indices,icustay_lengths=None,\
                       Data_Dir='./Sep_24_12_experiments_new/icustay_id',\
                       definition='t_sepsis_min'):
    
    
    icuid_sequence=np.load(Data_Dir+definition[1:]+'.npy')
        
    idx1_septic_original=np.where(icuid_sequence==idx1_septicicuid)[0][0]
    
    if icustay_lengths is not None:
        print('The length of current patient',icustay_lengths[idx1_septic_original])
        
    test_patients=np.concatenate(test_patient_indices)
    
    return np.where(test_patients==idx1_septic_original)[0][0]

def finding_sample_idxs(idx1_septicicuid,test_patient_indices,icustay_lengths=None,\
                       Data_Dir='./Sep_24_12_experiments_new/icustay_id',\
                       definition='t_sepsis_min'):
    
    
    icuid_sequence=np.load(Data_Dir+definition[1:]+'.npy')
        
    idx1_septic_original=np.array([np.where(icuid_sequence==idx1_septicicuid[i])[0][0]\
                                   for i in range(len(idx1_septicicuid))])
    
    if icustay_lengths is not None:
        print('The length of current patient',icustay_lengths[idx1_septic_original])
        
    test_patients=np.concatenate(test_patient_indices)
    
    return np.array([np.where(test_patients==idx1_septic_original[i])[0][0]\
                                   for i in range(len(idx1_septic_original))])


def tresholding(probs_sample,thred):
    
    idx=np.where(probs_sample>=thred)[0][0]
    return np.array([int(i>=idx) for i in range(len(probs_sample))])

def rect_line_at_turn(path,turn_value=1,replace_value=0):

    """
        change binary data such that when there is a turn, we have rectangle turn 
        
        for example, [0,0,1,1] with hidden time [0,1,2,3] will be transformed to [0,0,0,1,1] with time [0,1,2,2,3]
    return:
        new time seq [0,1,2,2,3], new path [0,0,0,1,1], original time seq [0,1,2,3] and old path (the input path)
        
    """
    num=len(path)
    
    time_seq=np.arange(num)
    
    try:
        repeated_idx=np.where(path==turn_value)[0][0]
    
        new_path=np.insert(path,repeated_idx,0)
        new_time_seq=np.insert(time_seq,repeated_idx,time_seq[repeated_idx])
    
        true_labels=np.zeros(len(new_path))
        true_labels[-1]=1
    
        true_times=np.arange(len(new_path))
        true_times[-1]=true_times[-2]
    
        return new_time_seq,new_path,true_times,true_labels
    except:
        
        return np.arange(len(path)),path,np.arange(len(path)),path
        
def trajectory_plot(probs_sample,labels_sample,thred=None,\
                    labels=['Risk score','Labels for T=6','Ground truth'],\
                    figsize=(10,3),fontsize=14,savename=None):
    
    plt.figure(figsize=figsize)

    new_time_seq,new_path,true_times,true_labels=rect_line_at_turn(labels_sample)
    
    plt.plot(true_times,true_labels,linestyle='-.',label='Ground truth',lw=2,color=sns.color_palette()[1])
    
    plt.plot(new_time_seq,new_path,lw=2,label='Labels for T=6',color=sns.color_palette()[1])
    
    plt.plot(probs_sample,linestyle=':',lw=1.5,marker='x',label='Risk score',color=sns.color_palette()[0])
    
    if thred is not None:
        
        threds=[thred for i in range(len(probs_sample))]
        
        plt.plot(threds,lw=2,linestyle=':',label='Threshold',color=sns.color_palette()[2])
        
        pred_labels=tresholding(probs_sample,thred)
        
        
        new_time_pred,pred_paths,_,_=rect_line_at_turn(pred_labels)

        plt.plot(new_time_pred,pred_paths,lw=2,label='Predicted labels',color=sns.color_palette()[2])

    plt.xlabel('ICU length-of-stay since ICU admission (Hour) of one septic patient',fontsize=fontsize)

    plt.legend(loc='upper left',bbox_to_anchor=(1.005, 1),fontsize=fontsize-1)

    plt.xticks(fontsize=fontsize-1)
    plt.yticks(fontsize=fontsize-1)
    
    if savename is not None:
        
        plt.savefig(savename+'.jpeg',dpi=350,bbox_inches='tight')
    else:
        plt.show()
        
