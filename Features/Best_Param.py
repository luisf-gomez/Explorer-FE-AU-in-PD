# -*- coding: utf-8 -*-
"""
@author: MSc. Luis Felipe Gómez Gómez - UAM - UdeA
"""

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, auc
from sklearn.svm import SVC


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

def metrics_resume(metrics,kernel,C,gamma):
    metrics_means=np.mean(metrics,axis=0)*100
    metrics_std=(np.std(metrics,axis=0))*100
    
    print('*'*50)
    print('Accuracy: %.1f $\pm$ %.1f'% (metrics_means[0],metrics_std[0]))
    print('Sensitivity: %.1f $\pm$ %.1f'% (metrics_means[1],metrics_std[1]))
    print('Specificity: %.1f $\pm$ %.1f'% (metrics_means[2],metrics_std[2]))
    print('F1_Score: %.1f $\pm$ %.1f'% (metrics_means[3],metrics_std[3]))
    
    
    print('%.1f $\pm$ %.1f \t %.1f $\pm$ %.1f \t %.1f $\pm$ %.1f \t %.1f $\pm$ %.1f'% 
          (metrics_means[0],metrics_std[0],metrics_means[1],metrics_std[1],
           metrics_means[2],metrics_std[2],metrics_means[3],metrics_std[3]))
    if (kernel == 'rbf'):
        print('$\mathcal{G}$: %.0e -- %.0e \t & \t'%(C,gamma),end='')
    elif (kernel == 'linear'):
        print('$\mathcal{L}$: %.0e \t & \t'%(C),end='')
    print('%.1f $\pm$ %.1f \t & \t %.1f $\pm$ %.1f \t & \t %.1f $\pm$ %.1f \t & \t %.1f $\pm$ %.1f'% 
          (metrics_means[0],metrics_std[0],metrics_means[1],metrics_std[1],
           metrics_means[2],metrics_std[2],metrics_means[3],metrics_std[3]))
    

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

def best_params(Test,n=72,ker='rbf'):

    kernels = []
    gamma_C = []
    for k in range(Test.shape[0]):
        if ker == 'rbf':
        
            qwe = Test[k][np.arange(n)*4+1].astype(float)
            idx = np.argwhere(qwe==np.max(qwe))
            kernels.append(Test[k,idx*4+2])
            gamma_C.append(np.char.add(np.char.add(Test[k,idx*4+3].astype(str),' - '),Test[k,idx*4+4].astype(str)))
        
        elif ker == 'linear':
            qwe = Test[k][np.arange(n,Test.shape[1]//4)*4+1].astype(float)
            idx = np.argwhere(qwe==np.max(qwe))+n
            kernels.append(Test[k,idx*4+2])
            gamma_C.append(Test[k,idx*4+3].astype(str))
        
    kernels=np.vstack(kernels)
    gamma_C=np.vstack(gamma_C)
    kernel = stats.mode(kernels)[0][0]
    if ker == 'rbf': 
        C, gamma = stats.mode(gamma_C)[0][0][0].split(' - ')
        pre_idx = np.roll(Test[0]==kernel,-1)*np.roll(Test[0]==C,-2)*np.roll(Test[0]==gamma,-3)
    elif ker == 'linear':
        C = stats.mode(gamma_C)[0][0][0].split(' - ')[0]
        pre_idx = np.roll(Test[0]==kernel,-1)*np.roll(Test[0]==C,-2)
        
    idx = np.argwhere(pre_idx)
    mean = np.mean(Test[:,idx].reshape(-1).astype(float))*100
    std = np.std(Test[:,idx].reshape(-1).astype(float))*100
    
    print('Kernel: %s, C: %s '%(stats.mode(kernels)[0][0],stats.mode(gamma_C)[0][0]))
    if ker == 'rbf':
        return kernel, C, gamma
    elif ker == 'linear':
        return kernel, C, 1
    
def metrics(sequence, Base, Modelo, optimal_parameters, Triplet=False):
    
    AUC_Vector=[]
    FNR_Vector=[]
    FPR_Vector=[]
    EER_Vector=[]
    histogramas_Vector,y_true_Vector=[],[]
    metrics=[]
    
    for i in range(5):
        
        # print('+'*50)
        if Triplet:
            X_train=np.load('./'+sequence+'/'+Base+'/'+Modelo+'/Data/X_train_Triplet_'+str(i)+'.npy')
            X_test=np.load('./'+sequence+'/'+Base+'/'+Modelo+'/Data/X_test_Triplet_'+str(i)+'.npy')

        else:
            X_train=np.load('./'+sequence+'/'+Base+'/'+Modelo+'/Data/X_train_Original_'+str(i)+'.npy')
            X_test=np.load('./'+sequence+'/'+Base+'/'+Modelo+'/Data/X_test_Original_'+str(i)+'.npy')
        
        y_train=np.load('./'+sequence+'/'+Base+'/'+Modelo+'/Data/y_train_'+str(i)+'.npy')
        y_test=np.load('./'+sequence+'/'+Base+'/'+Modelo+'/Data/y_test_'+str(i)+'.npy')

        
        weight={}
        for i in range(np.max(y_train)+1):
            weight[i]=np.sum(y_train==i)/len(y_train)
        weight
        
        mean=np.mean(X_train,axis=0)
        std=np.std(X_train,axis=0)+1e-23
        
        X_train=(X_train-mean)/std
        X_test=(X_test-mean)/std
        
        parameters=optimal_parameters[Modelo]
        kernel=parameters['kernel']
        C=parameters['C']
        gamma=parameters['gamma']
        clf = SVC(C=C,kernel=kernel,gamma=gamma,probability=True,class_weight=weight)
        clf.fit(X_train, y_train)
        # print()
        y_true, y_pred = y_test, clf.predict(X_test)
        # y_true, y_pred = y_train, clf.predict(X_train)

        # np.save('./Optimal/y_pred_'+Modelo+'_'+str(i)+'.npy',y_pred)        
        histogramas=clf.decision_function(X_test)
        # histogramas=clf.decision_function(X_train)
        histogramas_Vector.append(histogramas)
        y_true_Vector.append(y_true)
        FNR,FPR,AUC,EER=FNR_FPR(histogramas,y_true)
        AUC_Vector.append(AUC)
        FNR_Vector.append(FNR)
        FPR_Vector.append(FPR)
        EER_Vector.append(EER)
        # plt.subplot(122)
        # plt.plot(FPR,1-FNR)
        metrics.append(confusion_metrics(confusion_matrix(y_true, y_pred)))
        
    et=np.hstack(y_true_Vector)
    
    metrics_resume(metrics,kernel,C,gamma)
    print('EER: \t %.1f $\pm$ %.1f'% (np.mean(EER_Vector),np.std(EER_Vector)))
    
    print('*'*50)

def scan_Model(sequence, Base, Modelo, Triplet=False):

    print('-'*50)
    print()
    print('Parametros para el modelo',Modelo,'de',Base,':')
    print()
    
    if Triplet:
        Train = np.load('./'+sequence+'/'+Base+'/Performance_'+Modelo+'_train.npy')
        Test = np.load('./'+sequence+'/'+Base+'/Performance_'+Modelo+'_test.npy')
        
    else:
        Train = np.load('./'+sequence+'/'+Base+'/Performance_Original_'+Modelo+'_train.npy')
        Test = np.load('./'+sequence+'/'+Base+'/Performance_Original_'+Modelo+'_test.npy')

    n = np.argwhere(Test[0]=='linear')[0]//4
    if n != 0:
        kernel, C, gamma = best_params(Test,n = n,ker='rbf')
        optimal_parameters={}
        optimal_parameters[Modelo]={'kernel': kernel,
                                  'gamma': float(gamma),
                                  'C': float(C),}
        
        metrics(sequence, Base, Modelo, optimal_parameters, Triplet=Triplet)
        
    kernel, C, gamma = best_params(Test,n = n,ker='linear')
    
    optimal_parameters={}
    optimal_parameters[Modelo]={'kernel': kernel,
                              'gamma': float(gamma),
                              'C': float(C),}
    
    metrics(sequence, Base, Modelo, optimal_parameters, Triplet=Triplet)


#%%%
    
# =============================================================================
#  Experimentos 1 - VGGFace
# =============================================================================
Base, model_name = 'FA', 'Freeze_100'
for sequence in ['T1', 'T2', 'All']:
    scan_Model(sequence, Base, model_name, Triplet=False )

#%%%%

# =============================================================================
#  Experimento 2 - VGGFace + EmotioNet
# =============================================================================

Base = 'AU_PD'
for model_name in ['Freeze_75', 'Freeze_50', 'VGG8', 'ResNet7']:
    for sequence in ['T1', 'T2', 'All']:
        scan_Model( sequence, Base, model_name, Triplet=False )

#%%%%
        
# =============================================================================
#  Experimento 3 - VGGFace + EmotioNet + Triplet
# =============================================================================

Base = 'AU_PD'
for model_name in ['Freeze_75', 'Freeze_50', 'VGG8', 'ResNet7']:
    for sequence in ['T1', 'T2', 'All']:
        scan_Model( sequence, Base, model_name, Triplet=True )
