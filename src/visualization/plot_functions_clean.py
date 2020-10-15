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



############################ For boxplot ############################
            
def stacked_barplot3lists_compare(data1,data2, data3,\
                                  label1,label2,label3,\
                                  xlabel,ylabel, labels,\
                                  width=0.4,fontsize=14,\
                                  color=colors_barplot,\
                                  figsize=(15,8),save_name=None):
    """
    plot for stacked bars. We restrick to 3 data comparision.
    
    Input:
        data1,data2,data3 are three lists-data, each containing two sub-data:
                
                dataX=[dataX-subdata1,dataX-subdata2] for X in [1,2,3]
    
        label1,label2,label3 are three lists-labels,each containing two sub-list:
    
                labelX=[labelX-sublabel1,labelX-sublabel2] for X in [1,2,3]
                
        xlabel,ylabel, x/y-axis names
    
        labels: different xlabel values in data
        
        savetitle: if None: print plot; else: save to savetitle.png
        
    """

    sns.set(color_codes=True)
    x = np.arange(len(labels))  # the label locations

    
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width*2/3, data1[0], width/2, label=label1[0],hatch="X",color= colors_barplot[0])
    rects1_ = ax.bar(x - width*2/3, data1[1], width/2, bottom=data1[0],label=label1[1],color=colors_barplot[0])
    rects2 = ax.bar(x - width/6, data2[0], width/2, label=label2[0], hatch="X",color=colors_barplot[1])
    rects2_ = ax.bar(x - width/6, data2[1], width/2, bottom=data2[0], label=label2[1],color=colors_barplot[1])
    
    rects3 = ax.bar(x + width/3, data3[0], width/2, label=label3[0],hatch="X",color=colors_barplot[2])
    rects3_ = ax.bar(x+width/3, data3[1], width/2, bottom=data3[0], label=label3[1],color=colors_barplot[2]) 

    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel,fontsize=fontsize)

    ax.set_xticks(x)
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_xticklabels(labels,fontsize=fontsize)
    
    ax.tick_params(axis="x", labelsize=fontsize-3)
    ax.tick_params(axis="y", labelsize=fontsize-3)
    ax.legend(fontsize=fontsize,bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)


    fig.tight_layout()
    
    if save_name==None:
        plt.show()
    else:
        plt.savefig(save_name+".png",dpi=300)
  

############################ For confusion matrix plot ############################


def plot_confusions(cm, target_names,figsize=(5,3), normalised=False, fontsize=16,font_scale=1.0,savetitle=None):

    """
        confusion matrix (CM) plot (either orignal number or normalising along the row)
        
       
       Input:
       
        cm: the target 2-dim CM
        target_names: 2 class names
        fontsize: number size in CM
        font_scale: size of target_names
        savetitle: if None: print CM; else: save to save_name.png
    
    """
    fmt='g'
    
    if normalised:
        
        a=cm/np.repeat(cm.sum(axis=1),2).reshape(2,2)
        cm=a
        fmt='.2%'
    
    df_cm1 = pd.DataFrame(cm, target_names, target_names)
    
    fig, ax = plt.subplots(figsize=figsize) 
    sns.set(font_scale=font_scale)#for label size
    sns.heatmap(df_cm1, cmap="Blues", cbar=False, annot=True,annot_kws={"size": fontsize},fmt=fmt,ax=ax)# font size
    
    if save_name is not None:
        plt.savefig(save_name+'.png',dpi=300)         
    else:    
        
        plt.show() 


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
        plt.savefig(save_name+'.png',dpi=300)
    else:        
        plt.show()

        
def auc_subplots(trues_list,probs_list,names,\
                 fontsize=14,figsize=(15,5),\
                 titles=MODELS, colors=colors_auc,\
                 linestyles=linestyles,lw=2,\
                 loc="lower right",save_name=None):
    
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
        
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.title(titles[-1])
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    if save_name is not None:
        plt.savefig(save_name+'.png',dpi=300)
    else:
        
        plt.show()


        
def auc_plot_patient_level(fprs,tprs,names,fontsize=14,\
                            colors=colors_auc,titles=MODELS,\
                            linestyles=linestyles,lw = 2,\
                            loc="lower right",save_name=None):
    
    """
        AUC plots in one figure via computed fprs and tprs
        
    Input:
    
        fprs: fpr list for different sets of data
        
                eg, for 2 set of data, [[fpr for data set1],[fpr for data set2]]
                
        tprs: tpr list for different sets of data
        
                eg, for 2 set of data, [[tpr for data set1],[tpr for data set2]]

            
        names: curve labels
        
        save_name: if None: print figure; else: save to save_name.png

    
    
    """

    num=len(fprs)
    plt.figure()
    
    for i in range(num):
        
        roc_auc = auc(fprs[i], tprs[i])

        plt.plot(fprs[i], tprs[i], color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.ylabel('True Positive Rate',fontsize=fontsize)
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    if save_name is not None:
        plt.savefig(save_name+'.png',dpi=300)
    else:
        
        plt.show()

        
def auc_subplots_patient_level(fprs_list,tprs_list,names,\
                               fontsize=14,figsize=(15,5),\
                               colors=colors_auc,titles=MODELS,\
                               linestyles=linestyles,lw = 2,\
                               loc="lower right", save_name=None):
    """
        
        
        
        AUC plots for different models via computed fprs and tprs
        
    Input:
    
        fprs_list: list of fpr lists 
        
                eg, for three models for 2 set of data,[[model1 fpr-list], [model2 fpr-list], [model3 fpr-list]]
                    [model1 fpr-list]=[[fpr for data set1],[fpr for data set2]]
                
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
        
        roc_auc = auc(fprs_list[0][i], tprs_list[0][i])

        plt.plot(fprs_list[0][i], tprs_list[0][i], color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)

    
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.ylabel('True Positive Rate',fontsize=fontsize)
    plt.title(titles[0])
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    plt.subplot(132)
    
    num=len(tprs_list[1])    

    for i in range(num):
        
        roc_auc = auc(fprs_list[1][i], tprs_list[1][i])

        plt.plot(fprs_list[1][i], tprs_list[1][i], color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.title(titles[1])
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    
    plt.subplot(133)

    num=len(tprs_list[-1])    

    for i in range(num):
        
        roc_auc = auc(fprs_list[-1][i], tprs_list[-1][i])

        plt.plot(fprs_list[-1][i], tprs_list[-1][i], color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.title(titles[-1])
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    if save_name is not None:
        plt.savefig(save_name+'.png',dpi=300)
    else:
        
        plt.show()

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
        plt.savefig(save_name+'.png',dpi=300)
    else:
        
        plt.show()

        
def boxplots_prediction_time_inadvance(data_seqs,name_seq,ylabel,titles=MODELS,\
                                       figsize=(15,9),fontsize=14,notch=False,\
                                       widths=[0.7,0.7,0.7],savetitle=None):
    """
    
        Boxplots for predicted sepsis time in advance under different definitions for three models
        
    Input:
        
        data_seqs: [[data_seq for lgbm],[data_seq for lstm],[data_seq for coxphm]]
        
        for each model, we have data for three definitions:
        
            [data_seq for model]=[[data_seq for model and H1],[data_seq for model and H2],[data_seq for model and H3]]
        
        name_seq: names for three definition, name_seqs=['H1','H2','H3']
        
        ylabel: name for y axis
        
        save_name: if None: print figure; else: save to save_name.png

        
    """
    if widths==None:
        widths=[np.array([len(data_seqs[i][j]) for j in range(len(data_seqs[i]))]) for i in range(len(data_seqs))]
        widths_max=np.max(np.concatenate(widths))
        widths=widths/(widths_max*1.2)
        widths=[tuple(widths[i]) for i in range(len(widths))]
    
    y_lim=int(max(np.concatenate([np.concatenate(data_seqs[i]) for i in range(len(data_seqs))])))+1
    

    if y_lim>50:
        
        for i in range(len(data_seqs)):
            for j in range(len(data_seqs[i])):
                a=data_seqs[i][j]
                data_seqs[i][j]=a[np.where(a<50)[0]]
    
    y_lim=int(max(np.concatenate([np.concatenate(data_seqs[i]) for i in range(len(data_seqs))])))+1
        
        
    if y_lim>15:
        nn=5
    else:
        nn=3
    
    yticks=np.arange(round(y_lim/nn)+1)*nn
    
    if np.max(yticks)<=y_lim:
        yticks=np.arange(round(y_lim/nn)+2)*nn
        
    plt.figure(figsize=figsize)
    blue_circle = dict(markerfacecolor='b', marker='o')
    
    plt.subplot(131)
    plt.boxplot(data_seqs[0],flierprops=blue_circle,widths=widths[0],notch=notch)


    plt.ylabel(ylabel,fontsize=fontsize)
    plt.title(titles[0],fontsize=fontsize)
    plt.xticks(np.arange(len(data_seqs[0])+1)[1:],name_seq,fontsize=fontsize)
    plt.yticks(yticks,fontsize=fontsize)
    
    plt.subplot(132)    
    plt.boxplot(data_seqs[1],flierprops=blue_circle,widths=widths[1],notch=notch)


    plt.title(titles[1],fontsize=fontsize)
    plt.xticks(np.arange(len(data_seqs[1])+1)[1:],name_seq,fontsize=fontsize)
    plt.yticks(yticks,fontsize=fontsize)

    plt.subplot(133)    
    plt.boxplot(data_seqs[-1],flierprops=blue_circle,widths=widths[-1],notch=notch)


    plt.title(titles[-1],fontsize=fontsize)
    plt.xticks(np.arange(len(data_seqs[-1])+1)[1:],name_seq,fontsize=fontsize)
    plt.yticks(yticks,fontsize=fontsize)    
            
    if savetitle!=None:
        plt.savefig(savetitle+'.png',dpi=300)
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
        
def trajectory_plot(probs_sample,labels_sample,\
                    labels=['Risk score','Labels for T=6','Ground truth'],\
                    figsize=(10,3),fontsize=14,save_name=None):
    """
    
    """
               
    plt.figure(figsize=figsize)

    plt.plot(probs_sample,linestyle='-.',lw=2,label='Risk score')

    new_time_seq,new_path,true_times,true_labels=rect_line_at_turn(labels_sample)

    plt.plot(new_time_seq,new_path,lw=2,label='Labels for T=6')

    plt.plot(true_times,true_labels,linestyle=':',label='Ground truth',lw=2)

    plt.xlabel('ICU length-of-stay since ICU admission (Hour) of one spetic patient',fontsize=fontsize)

    plt.legend(loc='upper left',bbox_to_anchor=(1.005, 1),fontsize=fontsize-1)

    plt.xticks(fontsize=fontsize-1)
    plt.yticks(fontsize=fontsize-1)
    
    if save_name is not None:
        
        plt.savefig(save_name+'.png',dpi=300,bbox_inches='tight')
    else:
        plt.show()

def trajectory_plots(probs_samples,labels_sample, sublabels=MODELS,\
                    labels=['Risk score','Labels for T=6','Ground truth'],\
                    figsize=(10,3),fontsize=14,lw=2,save_name=None):
    """
        
        Pick one septic icu_id for illustration for three models.
        
        We look at the whole path of the patient till onset time, visualise the risk score at each time point, and thus predicted status, while comparing to the ground truth label.
        
    """    
    plt.figure(figsize=figsize)
    
    for i in range(len(sublabels)):
        
        plt.plot(np.arange(len(probs_samples[i])),probs_samples[i],linestyle='-.',\
                 lw=lw,label='Risk score for '+sublabels[i])

    new_time_seq,new_path,true_times,true_labels=rect_line_at_turn(labels_sample)

    plt.plot(new_time_seq,new_path,lw=lw,label='Labels for T=6')

    plt.plot(true_times,true_labels,linestyle=':',label='Ground truth',lw=lw)

    plt.xlabel('ICU length-of-stay since ICU admission (Hour) of one spetic patient',fontsize=fontsize)

    plt.legend(loc='upper left',bbox_to_anchor=(1.005, 1),fontsize=fontsize-1)

    plt.xticks(fontsize=fontsize-1)
    plt.yticks(fontsize=fontsize-1)
    
    if save_name is not None:
        
        plt.savefig(save_name+'.png',dpi=300,bbox_inches='tight')
    else:
        plt.show()




