#!/usr/bin/env python3 -W ignore::FutureWarning
# -*- coding: utf-8 -*-
"""
@author: MSc. Luis Felipe Gómez Gómez - UAM - UdeA
"""

from sklearn.svm  import SVC
import numpy as np
import time
import os

##############################################
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1000,100,10,1,1e-1,1e-2,1e-3, 1e-4],
                     'C': [1e-4, 1e-3, 1e-2, 1e-1,1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1e-4, 1e-3, 1e-2, 1e-1,1, 10, 100, 1000], 'gamma': [1]}]

directories = ['Freeze_100','Freeze_75','Freeze_50', 'VGG8', 'ResNet7']

data_accur=[]
data_accur2=[]

for path in directories:
    print('*'*50)
    print(path)

    AUC_Vector=[]
    FNR_Vector=[]
    FPR_Vector=[]
    EER_Vector=[]
    histogramas_Vector,y_true_Vector=[],[]
    metrics=[]
    data_accur=[]
    data_accur2=[]
    
    for i in range(5):
        
        print('+'*50)
        
        X_train=np.load('./'+path+'/Data/X_train_Triplet_'+str(i)+'.npy')
        X_test=np.load('./'+path+'/Data/X_test_Triplet_'+str(i)+'.npy')
        
        y_train=np.load('./'+path+'/Data/y_train_'+str(i)+'.npy')
        y_test=np.load('./'+path+'/Data/y_test_'+str(i)+'.npy')

        weight={}
        for i in range(np.max(y_train)+1):
            weight[i]=np.sum(y_train==i)/len(y_train)
        weight
        print(weight)
        
        mean=np.mean(X_train,axis=0)
        std=np.std(X_train,axis=0)+1e-23
        
        X_train=(X_train-mean)/std
        X_test=(X_test-mean)/std
        
        data=[]
        data.append(str(i)+'_fold')
        data2=[]
        data2.append(str(i)+'_fold')
        flag=0
        for parameters in tuned_parameters:
            for kernel in parameters['kernel']:
                for i,C in enumerate(parameters['C']):
                    for j,gamma in enumerate(parameters['gamma']):
                        # print("kernel: %s, C: %0.4f, gamma: %0.4f" %(kernel,C,gamma))
                        clf = SVC(C=C,kernel=kernel,gamma=gamma,probability=True,class_weight=weight)
                        clf.fit(X_train, y_train)
                        
                        y_true, y_pred = y_train, clf.predict(X_train)
                        i=str(i)
                        j=str(j)
                        folds=str(i)

                        data.append([clf.score(X_train, y_train),kernel,C,gamma])
                        data2.append([clf.score(X_test, y_test),kernel,C,gamma])
                        flag=flag+1
                        print(str(round(((flag)*100/46),2))+'%',end=' ')
        data=np.hstack(data)
        data_accur.append(data)
        data=np.hstack(data2)
        data_accur2.append(data)

    #%%%
    np.save('Performance_'+path+'_train',np.array(data_accur))
    np.save('Performance_'+path+'_test',np.array(data_accur2))

            
    q = 0
    means=np.mean(np.array(data_accur2)[q*5:5*(q+1),np.arange(data.shape[0]//4)*4+1].astype(float),axis=0)
    stds=np.std(np.array(data_accur2)[q*5:5*(q+1),np.arange(data.shape[0]//4)*4+1].astype(float),axis=0)
    idx=np.argmax(means)
    
    print('*'*50)
    print(path)
    print(round(100*means[idx],1),' +/- ',round(100*stds[idx],1))
    print(data_accur[0][idx*4+2:idx*4+5])

