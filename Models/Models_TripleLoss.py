# -*- coding: utf-8 -*-
"""
@author: MSc. Luis Felipe Gómez Gómez - UAM - UdeA
"""

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import  Model
from tensorflow.keras import backend as K

from matplotlib import pyplot
import tensorflow.keras.losses
import tensorflow.keras
import tensorflow as tf
from cv2 import cv2
import pandas as pd
import numpy as np
import random
import pickle
import sys
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
            #print(idx_train)
            
            for i in range(len(idx_1)):
                idx_test=np.hstack([idx_test,np.where(personas==idx_1[i])[0]])
            idx_test=idx_test.astype(int)
            #print(idx_test)
            #print(X_test)
            k.append([idx_train, idx_test])
    return idx_train, idx_test,k

def l2_norm(x):
    import tensorflow as tf
    #x2=tf.nn.l2_normalize(x,axis=None,epsilon=1e-12,name=None,dim=None)
    x2=tf.nn.l2_normalize(x,axis=1)
    return x2

#Generador de triplets
def triplet_generator(diccionarios, batch_size):
    len_dict = len(diccionarios)
    #Función que genera batches de triplets
    while True:

        anchor_matrix=[];
        pos_matrix=[];
        neg_matrix=[];
        #Generar tres batches con triplets aleatorios
        #x_anchor, x_positive, x_negative
        
        for dic in diccionarios:
            for i in range(batch_size//len_dict):
#                id_anchor = random.choice(list(dic.keys())
                id_anchor = random.choice(['HC','PD'])
#                id_anchor = random.choice(list(dic.keys())
                if id_anchor=='HC':
                    id_negative = 'PD'
                else:
                    id_negative = 'HC'
                while (id_anchor == id_negative):
                    id_negative = random.choice(list(dic.keys()))
                
                anchor = dic.get(id_anchor)
                negative = dic.get(id_negative)         
                
                emb_negative = random.choice(negative)
                [emb_anchor, emb_positive] = random.sample(anchor, 2)
                
                anchor_matrix.append(np.squeeze(np.expand_dims(emb_anchor, axis=0)))
                pos_matrix.append(np.squeeze(np.expand_dims(emb_positive, axis=0)))
                neg_matrix.append(np.squeeze(np.expand_dims(emb_negative, axis=0)))
                    
        x_anchor = np.asarray(anchor_matrix)
        x_positive = np.asarray(pos_matrix)
        x_negative = np.asarray(neg_matrix)

        # yield libera los datos que se generan en el generador
        #para que el algoritmo de entrenamiento los utilice
        yield ({'anchor_input': x_anchor,
               'positive_input': x_positive,
               'negative_input': x_negative},
               None)

def triplet_loss(inputs, margin='maxplus'):
    #Función de coste para el triplet, elegir entre: maxplus, softplus y paper
    alpha = 0.1
    anchor, positive, negative = inputs  
    
    anchor=tf.nn.l2_normalize(anchor,axis=1)
    positive=tf.nn.l2_normalize(positive,axis=1)
    negative=tf.nn.l2_normalize(negative,axis=1)    
    
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    
    positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
    negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    
    loss = (positive_distance**2) - (negative_distance**2) + alpha

    if margin == 'maxplus':
        loss = K.maximum(0.0, loss)# + (10 / negative_distance)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    elif margin == 'paper':
#        rho1 = 0
        rho2 = 0.01
        beta = 0.2
        loss = K.maximum(K.maximum(0.0,loss) - beta * negative_distance,rho2)
    return K.sum(loss) #Ojo lo he cambiado, he añadido sum por mean

def build_triplet_model(input_shape, embedding_model):
    # Genera el modelo triplet_loss a partir del modelo base (e.j. VGG)       
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    #Definimos la operacion a realizar, en este caso pasamos todas las entradas por el modelo suministrado
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    #Definimos entradas y salidas del modelo
    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]
    #Creamos el modelo
    triplet_model = Model(inputs, outputs)
    #Añadimos la funcion d eperdidas, podemos modificarla a nuestro gusto
    triplet_model.add_loss(K.sum(triplet_loss(outputs)))
    sgd = tensorflow.keras.optimizers.SGD(lr=1.25, decay=0, momentum=0.0, nesterov=True)    
    triplet_model.compile(loss=None,optimizer=sgd)

    return triplet_model

tensorflow.keras.losses.custom_loss=triplet_loss

path_images = '../Databases/FacePark/' # Facecrop images from Imagenet
path_labels = '../Databases/labels/FacePark_'
sequence2analysis = ['T1', 'T2', 'All']
AU_models = ['Freeze_75', 'Freeze_50', 'VGG8', 'ResNet7']

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
            
            for model_name in AU_models:
                #%%%  Loading models trained with EmotioNet

                model = tensorflow.keras.models.load_model('./AU/'+model_name+'/M1.h5')
                os.makedirs('../Features/'+sequence+'/AU_PD/'+model_name+'/Data/', exist_ok =True)
                os.makedirs('./Triplet/'+sequence+'/'+model_name+'/Model/', exist_ok =True)
                os.makedirs('./Triplet/'+sequence+'/'+model_name+'/Check/', exist_ok =True)

                #% Resnet50 
                if model_name.find('Freeze')>=0:

                    vgg_model = model.layers[0]
                    
                    temp_weights = [layer.get_weights() for layer in model.layers]
                    model2= Model(inputs=vgg_model.get_input_at(0), outputs=vgg_model.layers[-1].output)
                    model2=Dense(1024, activation='relu')(model2.layers[-1].input)
                    model2=Dense(8, activation='softmax')(model2)
                    model= Model(inputs=vgg_model.get_input_at(0), outputs=model2)
                    
                    for i in range(len(temp_weights)-1):
                        model.layers[-len(temp_weights)+i+1].set_weights(temp_weights[i+1]) 
                        
                    model2= Model(inputs=model.input, outputs=model.layers[-2].output)
                    model2=tensorflow.keras.layers.Reshape((2048,))(model2.layers[-1].input)
                    model2=Lambda(l2_norm)(model2)
                    
                    model= Model(inputs=model.input, outputs=model2)
                    model.summary()
        
                #% Tiny Resnet7
                elif model_name == 'ResNet7':

                    model2= Model(inputs=model.input, outputs=model.layers[-1].output)
                    model2=Lambda(l2_norm)(model2.layers[-1].input)
                    model= Model(inputs=model.input, outputs=model2)
                    model.summary()
        
                #% VGG8
                elif model_name == 'VGG8':

                    # model.pop() # 1024
                    # model.pop() # 512
                    # model.pop() # 256
                    model.pop() # 128
                    model.pop() # 64
                    model.pop() # 32
                    model.add(Lambda(l2_norm))
                    model.summary()      

    
                #%%%%%%  # Feature extraction for AU embeddings 

                model123=model
                X_train=model123.predict(images_list[idx_train],verbose = 1)
                X_test=model123.predict(images_list[idx_test],verbose = 1)
                
                X_train=X_train.reshape(-1,X_train.shape[-1])
                X_test=X_test.reshape(-1,X_test.shape[-1])

                
                np.save('../Features/'+sequence+'/AU_PD/'+model_name+'/Data/X_train_Original_'+str(numero),X_train)
                np.save('../Features/'+sequence+'/AU_PD/'+model_name+'/Data/X_test_Original_'+str(numero),X_test)
                
                #%%%% Training triplet loss models with PD information
                
                if K.image_data_format() == 'channels_first':
                    input_shape = (3, img_width, img_height)
                else:
                    input_shape = (img_width, img_height, 3)
                
                triplet_model = build_triplet_model(input_shape, model)
                triplet_model.summary()

                early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,verbose=0,restore_best_weights=True)
                checkpoint= tensorflow.keras.callbacks.ModelCheckpoint('checkpoint', monitor='val_loss', verbose=0, save_best_only=True)
                
                callbacks=[ early_stopping, checkpoint ]
                
                # Starting training

                nb_train_samples = 1000
                nb_test_samples = 500
                epochs = 200
                batch_size = 32
                history = triplet_model.fit_generator(triplet_generator(diccionarios_train, batch_size),
                                                            steps_per_epoch = nb_train_samples // batch_size,
                                                            validation_data=triplet_generator(diccionarios_test, batch_size),
                                                            validation_steps=nb_test_samples // batch_size,
                                                            epochs = epochs,                               
                                                            verbose = 1,
                                                            callbacks=callbacks)
            
                triplet_model.save('./Triplet/'+sequence+'/'+model_name+'/Model/M'+str(numero))
                triplet_model=tensorflow.keras.models.load_model('checkpoint',compile=False)
                triplet_model.save('./Triplet/'+sequence+'/'+model_name+'/Check/C'+str(numero))
                
                #%%%%%%  # Feature extraction for Triplet embeddings with PD information 

                model123=model
                X_train=triplet_model.predict([images_list[idx_train],images_list[idx_train],images_list[idx_train]])[0]
                X_test=triplet_model.predict([images_list[idx_test],images_list[idx_test],images_list[idx_test]])[0]
                
                X_train=X_train.reshape(-1,X_train.shape[-1])
                X_test=X_test.reshape(-1,X_test.shape[-1])    
                
                y_train=(patologics[idx_train]==np.array('PD'))*1
                y_test=(patologics[idx_test]==np.array('PD'))*1
                
                np.save('../Features/'+sequence+'/AU_PD/'+model_name+'/Data/X_train_Triplet_'+str(numero),X_train)
                np.save('../Features/'+sequence+'/AU_PD/'+model_name+'/Data/X_test_Triplet_'+str(numero),X_test)
                np.save('../Features/'+sequence+'/AU_PD/'+model_name+'/Data/users_train_CV'+str(numero),users[idx_train])
                np.save('../Features/'+sequence+'/AU_PD/'+model_name+'/Data/users_test_CV'+str(numero),users[idx_test])
                np.save('../Features/'+sequence+'/AU_PD/'+model_name+'/Data/y_train_'+str(numero),y_train)
                np.save('../Features/'+sequence+'/AU_PD/'+model_name+'/Data/y_test_'+str(numero),y_test)
                
