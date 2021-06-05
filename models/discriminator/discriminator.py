import tensorflow as tf

from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import PReLU
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.models import Model

import matplotlib.pyplot as plt
import numpy as np

LR = 24
HR = 96

def addMean(x):
    return x / 127.5 - 1

def delMean(x):
    return (x + 1) * 127.5


def DISCRIMINATOR(momentum=0.8):

    xInput  = Input((HR, HR, 3))
    x = Lambda(addMean)(xInput)
    
    #1
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(xInput)
    x = LeakyReLU(alpha=0.2)(x)

    #2
    x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)

    #3
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)

    #4
    x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)

    #5
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)

    #6
    x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)

    #7
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)

    #8
    x = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU()(x)
    x = Dense(1, activation='sigmoid')(x)

    x = Model(xInput, x)

    return x
    







