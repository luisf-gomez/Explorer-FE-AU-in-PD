# -*- coding: utf-8 -*-
"""
@author: MSc. Luis Felipe Gómez Gómez - UAM - UdeA
"""

import warnings
warnings.filterwarnings('ignore')

from keras_vggface.vggface import VGGFace
from tensorflow.keras import backend as K
import tensorflow.keras
import tensorflow as tf
from cv2 import cv2
import pandas as pd
import numpy as np
import random
import os

def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp

def split_users(personas,folds):
    unicos=np.unique(personas)
    random.shuffle(unicos)
    k=[]
    if folds==0:
        for i in range(len(unicos)):
            
            unicos=np.roll(unicos,1)
            idx_0=unicos[:-1]
            idx_1=unicos[-1:]
        
            idx_train=np.array([])
            idx_test=np.array([])
            for i in range(len(idx_0)):
                idx_train=np.hstack([idx_train,np.where(personas==idx_0[i])[0]])
            idx_train=idx_train.astype(int)
            #print(idx_train)
            
            for i in range(len(idx_1)):
                idx_test=np.hstack([idx_test,np.where(personas==idx_1[i])[0]])
            idx_test=idx_test.astype(int)
            #print(idx_test)
            #print(X_test)
            k.append([idx_train, idx_test])
        return idx_train, idx_test,k
    else:        
        for i in range(int(1/(1-folds/100))):
            unicos=np.roll(unicos,len(unicos)*(folds)//100)
            idx_0=unicos[:len(unicos)*(folds)//100]
            idx_1=unicos[len(unicos)*(folds)//100:]
        
            idx_train=np.array([])
            idx_test=np.array([])
            for i in range(len(idx_0)):
                idx_train=np.hstack([idx_train,np.where(personas==idx_0[i])[0]])
            idx_train=idx_train.astype(int)
            
            for i in range(len(idx_1)):
                idx_test=np.hstack([idx_test,np.where(personas==idx_1[i])[0]])
            idx_test=idx_test.astype(int)

            k.append([idx_train, idx_test])
    return idx_train, idx_test,k

def create_keys(df):
    df['emotion']
    df['label_emotion']

width,height = 224, 224

path_images = '../Databases/FacePark/' # Facecrop images from Imagenet
path_labels = '../Databases/labels/FacePark_'
sequence2analysis = [ 'Neutral', 'Onset', 'Apex', 'Offset', 'T1', 'T2', 'All']

with tf.device('/gpu:1'):
# if True:           
    
    # dimensions of our images.
    img_width, img_height = 224, 224
    
    K.clear_session()
    
    train_data_dir = path_images
    
    # Carga de imagenes
    images_list=[]
    lista=os.listdir(train_data_dir)
    lista.sort()
    for image in lista:
        img=cv2.imread(train_data_dir+image)
        img=cv2.resize(img,(img_width, img_height))
        images_list.append(img)
        
    images_list=np.array(images_list)
    images_list=images_list.astype('float32')
    images_list = preprocess_input(images_list)
    
    
    for sequence in sequence2analysis:
        path_label = path_labels + sequence + '.csv'

        df_all=pd.read_csv(path_label)
        
        gener=[]
        users=[]
        usersHC=[]
        usersPD=[]
        numberHC=[]
        numberPD=[]
        for number,name1 in enumerate(lista):
            name_short=name1[:5]
            users.append(name_short)
            if (name_short.rfind('HC'))>0:
                usersHC.append(name_short)
                numberHC.append(number)
            else:
                usersPD.append(name_short)
                numberPD.append(number)
        #    df_all['gener']=gener
        numberHC=np.array(numberHC)
        numberPD=np.array(numberPD)
        
        users=np.array(users)
        patologics=[]
        for patologic in users:
            patologics.append(patologic[-2:])
        patologics=np.array(patologics)
            
        idx_trainPD,idx_testPD,idx_all=split_users(np.array(usersPD),80)
        idx_trainHC,idx_testHC,idx_allHC=split_users(np.array(usersHC),80)
        data_accur=[]
        dict_results={}
        data_accur_ori=[]
        dict_results_ori={}
        for numero,(idx_trainPD,idx_testPD) in enumerate(idx_all):
            idx_trainHC,idx_testHC=idx_allHC[numero]
            idx_train=np.concatenate([numberHC[idx_trainHC],numberPD[idx_trainPD]])
            idx_test=np.concatenate([numberHC[idx_testHC],numberPD[idx_testPD]])
        
            print('##############################################')
            print('##############################################')
            print('###########   Iteraccion - ',numero,'   #############')
            print('##############################################')
            print('##############################################')  
            
            keys=patologics[idx_train]
            diccionarios_train=[]
            dic_train={}
            dic_train['HC']=[]
            dic_train['PD']=[]
            
            for (i,key) in enumerate(keys):
                dic_train[key].append(images_list[i])
            diccionarios_train.append(dic_train)
        
            keys=patologics[idx_test]
            diccionarios_test=[]
            dic_train={}
            dic_train['HC']=[]
            dic_train['PD']=[]
            for (i,key) in enumerate(keys):
                dic_train[key].append(images_list[i])
            diccionarios_test.append(dic_train)
            
            
            
            K.clear_session()

            os.makedirs('../Features/'+sequence+'/FA/Freeze_100/Data/', exist_ok =True)
            model = VGGFace(model='resnet50', include_top=False, input_shape=(width,height, 3), pooling='avg')
            
            model123=model
            X_train=model123.predict(images_list[idx_train],verbose = 1)
            X_test=model123.predict(images_list[idx_test],verbose = 1)
            
            X_train=X_train.reshape(-1,X_train.shape[-1])
            X_test=X_test.reshape(-1,X_test.shape[-1])
            
            y_train=(patologics[idx_train]==np.array('PD'))*1
            y_test=(patologics[idx_test]==np.array('PD'))*1

            
            np.save('../Features/'+sequence+'/FA/Freeze_100/Data/X_train_Original_'+str(numero),X_train)
            np.save('../Features/'+sequence+'/FA/Freeze_100/Data/X_test_Original_'+str(numero),X_test)
            np.save('../Features/'+sequence+'/FA/Freeze_100/Data/users_train_CV'+str(numero),users[idx_train])
            np.save('../Features/'+sequence+'/FA/Freeze_100/Data/users_test_CV'+str(numero),users[idx_test])
            np.save('../Features/'+sequence+'/FA/Freeze_100/Data/y_train_'+str(numero),y_train)
            np.save('../Features/'+sequence+'/FA/Freeze_100/Data/y_test_'+str(numero),y_test)
            
