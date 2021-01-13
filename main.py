import os
from models.generator.edsr import *
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image


def resolveImage(model, image):
    LR_batch = tf.expand_dims(image, axis=0)
    LR_batch = tf.cast(LR_batch, tf.float32)
    
    SR_batch = model(LR_batch)
    SR_batch = tf.clip_by_value(SR_batch, 0, 255)
    SR_batch = tf.round(SR_batch)
    SR_batch = tf.cast(SR_batch, tf.uint8)

    SR_image = SR_batch[0]
    return SR_image

def plotImages(HRmodel, SRmodel, imagePath):

    lr = np.array(Image.open(imagePath))
    if int(list(lr.shape)[-1]) != 3:
        print('Invalid Image')
        return
    
    hr = resolveImage(HRmodel, lr)
    sr = resolveImage(SRmodel, lr)

    print(f'Low resolution image: {lr.shape}')
    print(f'High resolution image: {hr.shape}')
    print(f'Super resolution image: {sr.shape}')
    
    
    plt.figure(figsize=(20, 20))
    
    plt.subplot(1, 3, 1)
    plt.title("Uploaded Low resolution image.")
    plt.imshow(lr)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.title("High resolution image using EDSR.\n(Note: GANs is not used to train this model.)")
    plt.imshow(hr)
    plt.xticks([])
    plt.yticks([])  

    plt.subplot(1, 3, 3)
    plt.title("Super resolution image using EDSR.")
    plt.imshow(sr)
    plt.xticks([])
    plt.yticks([])  

    plt.show()
        

DIR = './weights/article'
EDSR_HR = EDSR(factor=4, nResidualBlocks=16)
EDSR_HR.load_weights(os.path.join(DIR, 'weights-edsr-16-x4.h5'))

EDSR_SR = EDSR(factor=4, nResidualBlocks=16)
EDSR_SR.load_weights(os.path.join(DIR, 'weights-edsr-16-x4-fine-tuned.h5'))

plotImages(EDSR_HR, EDSR_SR, '0869x4-crop.png')

# for i in os.listdir('./images'):
#     print(i)
#     plotImages(EDSR_HR, EDSR_SR, f'./images/{i}')
#     print()

