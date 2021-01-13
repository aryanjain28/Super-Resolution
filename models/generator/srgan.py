import tensorflow as tf
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import PReLU 
from tensorflow.python.keras.layers import LeakyReLU

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras.models import Model

def addMean(x):
    return x / 255.0

def delMean(x):
    return (x + 1) * 127.5

def pixelShuffle(factor):
    return lambda x : tf.depth_to_space(x, factor)

def upsampleBlock(INPUT, filters):
    x = Conv2D(filters=filters, kernel_size=3, padding='same')(INPUT)
    x = Lambda(pixelShuffle(factor=2))(x)
    x = PReLU(shared_axes=[1,2])(x)
    return x


def redidualBlock(INPUT, filters, momentum=0.8):
    x = Conv2D(filters=filters, kernel_size=3, padding='same')(INPUT)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1,2])(x)

    x = Conv2D(filters=filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([INPUT, x])

    return x


def SRGAN(filters=64, nResidualBlocks=16):

    xInput = Input((None, None, 3))
    x = Lambda(addMean)(xInput)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = PReLU(shared_axes=[1,2])(x)
    R = x

    for _ in range(nResidualBlocks):
        x = redidualBlock(x, filters)

    x = Conv2D(filters=filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([R, x])

    x = upsampleBlock(x, filters * 4)
    x = upsampleBlock(x, filters * 4)

    x = Conv2D(filters=3, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(delMean)(x)

    return Model(xInput, x)
















