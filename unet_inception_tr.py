# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:31:17 2019

@author: hp
"""
    
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as Kb
from keras.layers import Concatenate, concatenate, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras.engine.input_layer import Input


smooth=1
def dice_coef(y_true, y_pred):
    y_true_f = Kb.flatten(y_true)
    y_pred_f = Kb.flatten(y_pred)
    intersection = Kb.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (Kb.sum(y_true_f*y_true_f) + Kb.sum(y_pred_f*y_pred_f) + smooth)



def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', kernel_initializer= 'he_normal')(x)
    conv_1x1 = BatchNormalization()(conv_1x1)
    conv_1x1 = Activation('relu')(conv_1x1)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', kernel_initializer= 'he_normal')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', kernel_initializer= 'he_normal')(conv_3x3)
    conv_3x3 = BatchNormalization()(conv_3x3)
    conv_3x3 = Activation('relu')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', kernel_initializer= 'he_normal')(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer= 'he_normal')(conv_5x5)
    conv_5x5 = BatchNormalization()(conv_5x5)
    conv_5x5 = Activation('relu')(conv_5x5)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', kernel_initializer= 'he_normal')(pool_proj)
    pool_proj = BatchNormalization()(pool_proj)
    pool_proj = Activation('relu')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

def unet_inception_tr(pretrained_weights = None,input_size = (240,240,3)):
    inputs = Input(input_size)
    
    inception1_1 = inception_module(inputs,
                     filters_1x1=16,
                     filters_3x3_reduce=8,
                     filters_3x3=16,
                     filters_5x5_reduce=8,
                     filters_5x5=16,
                     filters_pool_proj=16,
                      name = "inception1")
    
    inception1_2 = inception_module(inception1_1,
                     filters_1x1=16,
                     filters_3x3_reduce=8,
                     filters_3x3=16,
                     filters_5x5_reduce=8,
                     filters_5x5=16,
                     filters_pool_proj=16,
                      )
    concat_up_1 = Conv2D(16, (3,3), padding = 'same', kernel_initializer= 'he_normal')(inception1_2)
    concat_up_1 = BatchNormalization()(concat_up_1)
    concat_up_1 = Activation('relu')(concat_up_1)
    
    pool1 = MaxPooling2D(pool_size = (2,2))(inception1_2)
    
    
    inception2_1 = inception_module(pool1,
                     filters_1x1=32,
                     filters_3x3_reduce=16,
                     filters_3x3=32,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                      )
    
    inception2_2 = inception_module(inception2_1,
                     filters_1x1=32,
                     filters_3x3_reduce=16,
                     filters_3x3=32,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                      )
    concat_up_2 = Conv2D(32, (3,3), padding = 'same', kernel_initializer= 'he_normal')(inception2_2)
    concat_up_2 = BatchNormalization()(concat_up_2)
    concat_up_2 = Activation('relu')(concat_up_2)
    
    pool2 = MaxPooling2D(pool_size = (2,2))(inception2_2)
    
    
    inception3_1 = inception_module(pool2,
                     filters_1x1=64,
                     filters_3x3_reduce=32,
                     filters_3x3=64,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                      )
    
    inception3_2 = inception_module(inception3_1,
                     filters_1x1=64,
                     filters_3x3_reduce=32,
                     filters_3x3=64,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                      )
    concat_up_3 = Conv2D(64, (3,3), padding = 'same', kernel_initializer= 'he_normal')(inception3_2)
    concat_up_3 = BatchNormalization()(concat_up_3)
    concat_up_3 = Activation('relu')(concat_up_3)
    
    pool3 = MaxPooling2D(pool_size = (2,2))(inception3_2)
    
    
    inception4_1 = inception_module(pool3,
                     filters_1x1=128,
                     filters_3x3_reduce=64,
                     filters_3x3=128,
                     filters_5x5_reduce=64,
                     filters_5x5=128,
                     filters_pool_proj=128,
                      )
    
    inception4_2 = inception_module(inception4_1,
                     filters_1x1=128,
                     filters_3x3_reduce=64,
                     filters_3x3=128,
                     filters_5x5_reduce=64,
                     filters_5x5=128,
                     filters_pool_proj=128,
                      )
    drop4 = Dropout(0.5)(inception4_2)
    
    concat_up_4 = Conv2D(128, (3,3), padding = 'same', kernel_initializer= 'he_normal')(drop4)
    concat_up_4 = BatchNormalization()(concat_up_4)
    concat_up_4 = Activation('relu')(concat_up_4)
    
    
    
    pool4 = MaxPooling2D(pool_size = (2,2))(drop4)
    
    
    inception5_1 = inception_module(pool4,
                     filters_1x1=256,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=128,
                     filters_5x5=256,
                     filters_pool_proj=256,
                      )
    
    inception5_2 = inception_module(inception5_1,
                     filters_1x1=256,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=128,
                     filters_5x5=256,
                     filters_pool_proj=256,
                      )
    
    drop5 = Dropout(0.5)(inception5_2)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([concat_up_4,up6], axis = 3)
    
    inception6_1 = inception_module(merge6,
                     filters_1x1=128,
                     filters_3x3_reduce=64,
                     filters_3x3=128,
                     filters_5x5_reduce=64,
                     filters_5x5=128,
                     filters_pool_proj=128,
                      )
    
    inception6_2 = inception_module(inception6_1,
                     filters_1x1=128,
                     filters_3x3_reduce=64,
                     filters_3x3=128,
                     filters_5x5_reduce=64,
                     filters_5x5=128,
                     filters_pool_proj=128,
                      )
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(inception6_2))
    merge7 = concatenate([concat_up_3,up7], axis = 3)
    
    inception7_1 = inception_module(merge7,
                     filters_1x1=64,
                     filters_3x3_reduce=32,
                     filters_3x3=64,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                      )
    
    inception7_2 = inception_module(inception7_1,
                     filters_1x1=64,
                     filters_3x3_reduce=32,
                     filters_3x3=64,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                      )
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', 
                 kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(inception7_2))
    merge8 =concatenate([concat_up_2,up8],axis = 3)
    
    inception8_1 = inception_module(merge8,
                     filters_1x1=32,
                     filters_3x3_reduce=16,
                     filters_3x3=32,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                      )
    
    inception8_2 = inception_module(inception8_1,
                     filters_1x1=32,
                     filters_3x3_reduce=16,
                     filters_3x3=32,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                      )
    
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(inception8_2))
    merge9 = concatenate([concat_up_1,up9],axis = 3)
    
    inception9_1 = inception_module(merge9,
                     filters_1x1=16,
                     filters_3x3_reduce=8,
                     filters_3x3=16,
                     filters_5x5_reduce=8,
                     filters_5x5=16,
                     filters_pool_proj=16,
                      )
    
    inception9_2 = inception_module(inception9_1,
                     filters_1x1=16,
                     filters_3x3_reduce=8,
                     filters_3x3=16,
                     filters_5x5_reduce=8,
                     filters_5x5=16,
                     filters_pool_proj=16,
                      )
    inception9_3 = inception_module(inception9_2,
                     filters_1x1=4,
                     filters_3x3_reduce=2,
                     filters_3x3=4,
                     filters_5x5_reduce=2,
                     filters_5x5=4,
                     filters_pool_proj=4,
                      )
    
    conv10 = Conv2D(4, 1, activation = 'softmax')(inception9_3)
    

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy',dice_coef])
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model