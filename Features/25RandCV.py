#!/usr/bin/env python3 -W ignore::FutureWarning
# -*- coding: utf-8 -*-
"""
@author: MSc. Luis Felipe Gómez Gómez - UAM
"""


from tokenize import Triple

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.svm import SVC

import scipy.stats as stats
import numpy as np

import os



def stratified_group_k_fold(X, y, groups, k, seed=None):
    
    from collections import Counter, defaultdict
    import random
    
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def FNR_FPR(Datos,Etiquetas):
    Min=-5
    Max=5
    Dif=(Max-Min)/10
    umbral=np.arange(Min-Dif,Max+Dif,Dif/10)

    FPR=[]
    FNR=[]

    for i in umbral:
        Eti=Datos>i
        Eti=Eti*1
        tn,fp,fn,tp=confusion_matrix(Etiquetas,Eti).ravel()
        FPR=np.append(FPR,float(fp)/float(fp+tn))
        FNR=np.append(FNR,1-float(tp)/float(tp+fn))
        
    dif=abs(FPR-FNR)
    idx=np.where(dif==min(dif))[0]
    EER1=FPR[idx[0]]*100
    EER2=FNR[idx[0]]*100
    EER=(EER1+EER2)/2
    EER=round(EER,3)    
    AUC=round(auc(FPR,1-FNR),3)

    return (FNR,FPR,AUC,EER)

def metrics_resume(metrics):
    metrics_means=np.mean(metrics,axis=0)*100
    metrics_std=(np.std(metrics,axis=0))*100
    
    print('-'*50)
    print('Accuracy: %.1f +/- %.1f'% (metrics_means[0],metrics_std[0]))
    print('Sensitivity: %.1f +/- %.1f'% (metrics_means[1],metrics_std[1]))
    print('Specificity: %.1f +/- %.1f'% (metrics_means[2],metrics_std[2]))
    print('F1_Score: %.1f +/- %.1f'% (metrics_means[3],metrics_std[3]))
    

def confusion_metrics (conf_matrix):# save confusion matrix and slice into four pieces    
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]    
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
        
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TN / float(TN + FP))    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    
    return np.array([conf_accuracy,conf_sensitivity,conf_specificity,conf_f1])

def t_stats(data,label,p_value=0.01,num_feat=None):
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    
    data=((data-mean)/std)
    dicc=[]
    
    for m in range(data.shape[1]):
        p_k=[]
        t_k=[]
        for k,i in enumerate(np.unique(label)):
            for j in np.unique(label)[k+1:]: 
                t,p=stats.ttest_ind(data[label==i,m],data[label==j,m],equal_var=False)
                
                print("Welch’s t-test: statics - pvalue: \t %.2e \t - \t %.2e" %(t,p))
                t,p=stats.mannwhitneyu(data[label==i,m],data[label==j,m])
                
                print("Mann-Whitney U: statics - pvalue: \t %.2e \t - \t %.2e" %(t,p)) 


##############################################

optimal_parameters={}

BestModels = [ 
            ['Onset', 'FA', 'Freeze_100', False, 'linear', 1e-2, 1],
            ['All', 'FA', 'Freeze_100', False, 'linear', 1e-3, 1],

            ['All', 'AU_PD', 'Freeze_75', False, 'linear', 1e-1, 1],
            ['T1', 'AU_PD', 'Freeze_50', False, 'rbf', 1e+2, 1e-4],
            ['T2', 'AU_PD', 'VGG8', False, 'linear', 1e-2, 1],
            ['All', 'AU_PD', 'ResNet7', False, 'rbf', 1e+3, 1e-4],

            ['All', 'AU_PD', 'Freeze_75', True, 'linear', 1e-1, 1],
            ['T1', 'AU_PD', 'Freeze_50', True, 'linear', 1e-1, 1],
            ['T1', 'AU_PD', 'VGG8', True, 'linear', 1e-2, 1],
            ['T1', 'AU_PD', 'ResNet7', True, 'linear', 1e-1, 1],
        ]

for Sequence, Type, Model, TripletFlag, kernel, C, gamma in BestModels:
        
    optimal_parameters[Model]={'kernel': kernel,
                                'gamma': float(gamma),
                                'C': float(C),}

    AUC_Vector=[]
    FNR_Vector=[]
    FPR_Vector=[]
    EER_Vector=[]
    histogramas_Vector,y_true_Vector=[],[]
    metrics=[]
    
    for i in range(5):
        
        if TripletFlag:
            X_train=np.load('./'+Sequence+'/'+Type+'/'+Model+'/Data/X_train_Triplet_'+str(i)+'.npy')
            X_test=np.load('./'+Sequence+'/'+Type+'/'+Model+'/Data/X_test_Triplet_'+str(i)+'.npy')
        
        else:
            X_train=np.load('./'+Sequence+'/'+Type+'/'+Model+'/Data/X_train_Original_'+str(i)+'.npy')
            X_test=np.load('./'+Sequence+'/'+Type+'/'+Model+'/Data/X_test_Original_'+str(i)+'.npy')
            
        y_train=np.load('./'+Sequence+'/'+Type+'/'+Model+'/Data/y_train_'+str(i)+'.npy')
        y_test=np.load('./'+Sequence+'/'+Type+'/'+Model+'/Data/y_test_'+str(i)+'.npy')
        
        users_train=np.load('./'+Sequence+'/'+Type+'/'+Model+'/Data/users_train_CV'+str(i)+'.npy')
        users_test=np.load('./'+Sequence+'/'+Type+'/'+Model+'/Data/users_test_CV'+str(i)+'.npy')
        
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        users = np.hstack([users_train, users_test])
        
        for toca in range(5):
            
            for train_index, test_index in stratified_group_k_fold(X, y, users,k=5):
                users_train, users_test = users[train_index], users[test_index]
                y_train, y_test = y[train_index], y[test_index]
                X_train, X_test = X[train_index], X[test_index]
            
                weight={}
                for i in range(np.max(y_train)+1):
                    weight[i]=np.sum(y_train==i)/len(y_train)
                weight
                
                mean=np.mean(X_train,axis=0)
                std=np.std(X_train,axis=0)+1e-23
                
                X_train=(X_train-mean)/std
                X_test=(X_test-mean)/std
                
                data=[]
                parameters=optimal_parameters[Model]
                kernel=parameters['kernel']
                C=parameters['C']
                gamma=parameters['gamma']
                
                clf = SVC(C=C,kernel=kernel,gamma=gamma,probability=True,class_weight=weight)
                clf.fit(X_train, y_train)
                # print()
                y_true, y_pred = y_test, clf.predict(X_test)
                # y_true, y_pred = y_train, clf.predict(X_train)
        
                histogramas=clf.decision_function(X_test)
                # histogramas=clf.decision_function(X_train)
                histogramas_Vector.append(histogramas)
                y_true_Vector.append(y_true)
                
                FNR,FPR,AUC,EER=FNR_FPR(histogramas,y_true)
                AUC_Vector.append(AUC)
                FNR_Vector.append(FNR)
                FPR_Vector.append(FPR)
                EER_Vector.append(EER)
                metrics.append(confusion_metrics(confusion_matrix(y_true, y_pred)))
        
    print('*'*50)
    et=np.hstack(y_true_Vector)
    
    print(Sequence, Type, Model, TripletFlag)
    metrics_resume(metrics)
    print('EER: \t %.1f +/- %.1f'% (np.mean(EER_Vector),np.std(EER_Vector)))
    
    print('*'*50)
    
    t_stats(np.hstack(histogramas_Vector).reshape(-1,1),et)
    
    
    final_acc = np.array(metrics)[:,0]
    if TripletFlag:
        np.save('./StatisticalTest/'+Type+'_'+Model+'_Triplet', final_acc)
    else:
        np.save('./StatisticalTest/'+Type+'_'+Model+'_Original', final_acc)