import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def unet(pretrained_weights = None,input_size = (256,256,2), output_channels = 2, activation = 'relu', init_layers = 64):
    layers = init_layers
    
    inputs = Input(input_size)
    conv1 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    #conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    layers *= 2

    conv2 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    layers *= 2

    conv3 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    layers *= 2

    conv4 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    layers *= 2

    conv5 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #conv5 = BatchNormalization()(conv5)

    layers //= 2

    up6 = Conv2D(layers, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    #up6 = BatchNormalization()(up6)
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = BatchNormalization()(conv6)

    layers //= 2

    up7 = Conv2D(layers, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    #up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)

    layers //= 2

    up8 = Conv2D(layers, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
   # up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #conv8 = BatchNormalization()(conv8)

    layers //= 2
    
    up9 = Conv2D(layers, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    #conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(layers, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = BatchNormalization()(conv9)
    
    # Don't run this
    #conv9 = Conv2D(2, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(output_channels, 1, activation='linear')(conv9)
    #conv10 = BatchNormalization()(conv10)

    model = Model(inputs,conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss = "huber_loss", metrics=['mean_squared_error'])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


