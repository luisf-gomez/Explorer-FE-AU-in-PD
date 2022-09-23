# -*- coding: utf-8 -*-
"""
@author: MSc. Luis Felipe Gómez Gómez - UAM - UdeA
"""
#%%%

from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D, ZeroPadding2D, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K 
import matplotlib.pyplot as plt   
import tensorflow.keras
import numpy as np
import os

from keras_vggface.utils import preprocess_input
import pandas as pd

K.clear_session()

def identity_block(input_tensor, kernel_size, filters):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
    """
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), use_bias=False)(input_tensor)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """

    filters1, filters2, filters3 = filters


    x = Conv2D(filters1, (1, 1), use_bias=False,)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = Conv2D(filters2, kernel_size, strides=strides, padding='same',
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False)(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=False)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add([x, shortcut])
    x = Activation('relu')(x)
    return x
#%%%

width,height=224,224
names=['AU_1', 'AU_2', 'AU_4', 'AU_5', 'AU_6', 'AU_12', 'AU_25', 'AU_26']
path_labels = '../Databases/labels/Emotionet_labels.csv' # Use a custom version of the labels 
path_images = '../Databases/EmotioNet/' # Facecrop images from Imagenet

Model_name = 'ResNet7'
os.makedirs('./AU/'+Model_name+'/', exist_ok =True)
#%%%

# Model to train

img_input = Input(shape=(width,height, 3))

# Conv1 (7x7,64,stride=2)
x = ZeroPadding2D(padding=(3, 3))(img_input)

x = Conv2D(32, (7, 7),
                  strides=(2, 2),
                  padding='valid', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = ZeroPadding2D(padding=(1, 1))(x)

# 3x3 max pool,stride=2
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

# Conv2_x

# 1×1, 64
# 3×3, 64
# 1×1, 256

x = conv_block(x, 3, [32, 32, 128], strides=(1, 1))
x = identity_block(x, 3, [32, 32, 128])
x = identity_block(x, 3, [32, 32, 128])

# Conv3_x
#
# 1×1, 128
# 3×3, 128
# 1×1, 512

x = conv_block(x, 3, [64, 64, 256])
x = identity_block(x, 3, [64, 64, 256])
x = identity_block(x, 3, [64, 64, 256])
x = identity_block(x, 3, [64, 64, 256])


# average pool, 1000-d fc, sigmoid
x = GlobalAveragePooling2D()(x)
x = Dense(len(names), activation='sigmoid')(x)

# Create model.
model = Model(img_input, x)
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

#%%%%

df=pd.read_csv(path_labels)
batch_size = 128


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

model.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])

history =model.fit_generator(generator=train_generator,
                             steps_per_epoch=STEP_SIZE_TRAIN,
                             validation_data=valid_generator,
                             validation_steps=STEP_SIZE_VALID,
                             epochs=200,
                             callbacks=callbacks
)

model.save('./AU/'+Model_name+'/M1.h5')
