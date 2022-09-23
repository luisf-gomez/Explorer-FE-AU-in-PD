# -*- coding: utf-8 -*-
"""
@author: MSc. Luis Felipe Gómez Gómez - UAM - UdeA
"""
#%%%

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from tensorflow.keras import backend as K 
K.clear_session()


width,height=224,224
names=['AU_1', 'AU_2', 'AU_4', 'AU_5', 'AU_6', 'AU_12', 'AU_25', 'AU_26']
path_labels = '../Databases/labels/Emotionet_labels.csv' # Use a custom version of the labels 
path_images = '../Databases/EmotioNet/' # Facecrop images from Imagenet

Model_name = 'VGG8'
os.makedirs('./AU/'+Model_name+'/', exist_ok =True)

#%%%

# Model to train

num_features = 16
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 3), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dense(len(names), activation='sigmoid'))

model.summary()

#%%%


df=pd.read_csv(path_labels)
batch_size = 128


train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, 
        rotation_range=10,
        )

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
        directory=path_images,
        dataframe=df[:469000],
        target_size=(width, height),
        batch_size=batch_size,
        x_col='name',
        y_col=names,
        class_mode='other',
        shuffle=True)

valid_generator = test_datagen.flow_from_dataframe(
        directory=path_images,
        dataframe=df[469000:536000],
        target_size=(width, height),
        x_col='name',
        y_col=names,
        batch_size=batch_size,
        shuffle=True,
        class_mode='other')

test_generator = test_datagen.flow_from_dataframe(
        directory=path_images,
        dataframe=df[536000:],
        target_size=(width, height),
        x_col='name',
        y_col=names,
        batch_size=1,
        class_mode='other',shuffle=False)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint('checkpoint', monitor='val_loss', verbose=0, save_best_only=True)
callbacks=[ early_stopping, checkpoint ]


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),loss="binary_crossentropy",metrics=["accuracy"])

history =model.fit_generator(generator=train_generator,
                             steps_per_epoch=STEP_SIZE_TRAIN,
                             validation_data=valid_generator,
                             validation_steps=STEP_SIZE_VALID,
                             epochs=200,
                             callbacks=callbacks
)

model.save('./AU/'+Model_name+'/M1.h5')
