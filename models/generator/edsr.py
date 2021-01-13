
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Input

from tensorflow.python.keras.models import Model


RGB_MEAN = np.array([[0.4488, 0.4371, 0.4040]]) * 255

def meanAdd(x):
    return (x-RGB_MEAN) / 127.5

def meanSubtract(x):
    return (x*127.5) + RGB_MEAN

def pixelShuffle(scale):
    return lambda x : tf.nn.depth_to_space(x, scale)


def EDSR(nFilters=64, nResidualBlocks=16, factor=4):

    xInput = Input((None, None, 3))
    x = Lambda(meanAdd)(xInput)

    x = Conv2D(nFilters, kernel_size=3, padding='same')(x)
    R = x

    for _ in range(nResidualBlocks):
        R = residualBlock(R, nFilters)

    R = Conv2D(nFilters, kernel_size=3, padding='same')(R)
    x = Add()([x, R])

    x = upsamplingBlock(x, nFilters, factor)
    x = Conv2D(3, kernel_size=3, padding='same')(x)

    x = Lambda(meanSubtract)(x)
    return Model(xInput, x, name='EDSR')


def residualBlock(INPUT, f):
    x = Conv2D(f, kernel_size=3, padding='same', activation='relu')(INPUT)
    x = Conv2D(f, kernel_size=3, padding='same')(x)
    x = Add()([INPUT, x])
    return x

def upsamplingBlock(INPUT, f, factor):

    if factor==2:
        x = Conv2D(f*(factor**2), kernel_size=3, padding='same')(INPUT)

    elif factor==3:
        x = Conv2D(f*(factor**2), kernel_size=3, padding='same')(INPUT)

    elif factor==4:
        factor=2
        x = Conv2D(f*(factor**2), kernel_size=3, padding='same')(INPUT)
        x = Lambda(pixelShuffle(scale=factor))(x)
        x = Conv2D(f*(factor**2), kernel_size=3, padding='same')(x)
        
    x = Lambda(pixelShuffle(scale=factor))(x)
    return x





