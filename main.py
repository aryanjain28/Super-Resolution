import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from models.generator.edsr import *
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

import numpy as np

from tkinter import *
from tkinter.font import Font
import PIL.ImageTk as ImageTk
from PIL import Image
from tkinter import filedialog, messagebox


def resolveImage(model, image):
    LR_batch = tf.expand_dims(image, axis=0)
    LR_batch = tf.cast(LR_batch, tf.float32)
    
    SR_batch = model(LR_batch)
    SR_batch = tf.clip_by_value(SR_batch, 0, 255)
    SR_batch = tf.round(SR_batch)
    SR_batch = tf.cast(SR_batch, tf.uint8)

    SR_image = SR_batch[0]
    return SR_image

def getImages(imagePath):

    lr = np.array(Image.open(imagePath))

    if int(list(lr.shape)[-1]) != 3:
        messagebox.showerror(title="Invalid file type", message=f"Image Dimension : {lr.shape}")
        print('Invalid Image')
        return True
    
    hr = resolveImage(HRmodel, lr)
    sr = resolveImage(SRmodel, lr)

    print(f'Low resolution image: {lr.shape}')
    print(f'High resolution image: {hr.shape}')
    print(f'Super resolution image: {sr.shape}')
    print("___________________________________")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Uploaded Low resolution image.\nShape: {lr.shape[0]}x{lr.shape[1]}x{lr.shape[2]}")
    plt.imshow(lr)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.title(f"Generated High resolution image.\nShape: {hr.shape[0]}x{hr.shape[1]}x{hr.shape[2]}")
    plt.imshow(hr)
    plt.xticks([])
    plt.yticks([])  

    plt.subplot(1, 3, 3)
    plt.title(f"Fine tuned Super resolution image.\nShape: {sr.shape[0]}x{sr.shape[1]}x{sr.shape[2]}")
    plt.imshow(sr)
    plt.xticks([])
    plt.yticks([])  

    plt.savefig('./finalImages/superResolutionImage.png')

    return False


def resetPanel():
    panel.config(image='')
    panel.config(text="Click The Button Below")
    panel.config(font=Font(size=60))

    root.update()


def showImage(root, imagePath=None):
    global img

    panel.config(image='')
    panel.config(text='Loading results...')
    panel.config(font=Font(size=20))

    root.update()

    invalid = getImages(imagePath=imagePath)
    if(invalid == True):
        resetPanel()
        return

    img = Image.open('./finalImages/superResolutionImage.png')

    WIDTH = img.width
    HEIGHT = img.height

    NEW_WIDTH = 1600
    NEW_HEIGHT = 600

    if (WIDTH > HEIGHT):
        
        HEIGHT = int((NEW_WIDTH*HEIGHT)/WIDTH)
        WIDTH = NEW_WIDTH
        
        if(HEIGHT > NEW_HEIGHT):
            WIDTH = int((NEW_HEIGHT*WIDTH)/HEIGHT)
            HEIGHT = NEW_HEIGHT

    else:

        WIDTH = int((NEW_HEIGHT*WIDTH)/HEIGHT)
        HEIGHT = NEW_HEIGHT

        if(WIDTH >= NEW_WIDTH):
            HEIGHT = int((NEW_WIDTH*HEIGHT)/WIDTH)
            WIDTH = NEW_WIDTH

    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img, CENTER)

    panel.place(relx=0.5, rely=0.46, anchor=CENTER)    
    panel.config(image=img)
    panel.config(text=None)
    panel.image = img

    root.update()



def uploadImage():
    global panel, initialState, root

    initialState=True
    img = None

    def open_img():
        initialState = False
        x = filedialog.askopenfilename(title ='pen')
        extension = x.split(".")[-1].lower()

        if(extension == 'png' or extension == 'jpeg' or extension == 'jpg'):
            showImage(root, x)
        else:
            messagebox.showerror(title="Invalid file type", message=f"Please select an image (File type: {extension})")
            resetPanel()

    root = Tk()
    root.title("Low Resolution Image to Super Resolution Image")
    root.geometry("1300x700+25+25")
    root.configure(bg="#243142")
    root.resizable(width = True, height = True)

    textMsg="Click The Button Below" 
    btnText="Choose File"
    img=None

    panel = Label(root, text=textMsg, image=img, bg="#243142", fg="#CDCDCD", font=Font(family="Helvetica", size=60))
    panel.place(relx=0.5, rely=0.46, anchor=CENTER)

    btn = Button(root, text=btnText, command = open_img, height=1, width=20, bg="#C74545", fg="white", font=Font(family="Helvetica", size=13, weight="bold"))
    btn.place(relx=0.5, rely=0.95, anchor=CENTER)

    root.mainloop()


DIR = './weights/article'
EDSR_HR = EDSR(factor=4, nResidualBlocks=16)
EDSR_HR.load_weights(os.path.join(DIR, 'weights-edsr-16-x4.h5'))

EDSR_SR = EDSR(factor=4, nResidualBlocks=16)
EDSR_SR.load_weights(os.path.join(DIR, 'weights-edsr-16-x4-fine-tuned.h5'))


global HRmodel
global SRmodel

HRmodel = EDSR_HR
SRmodel = EDSR_SR

uploadImage()