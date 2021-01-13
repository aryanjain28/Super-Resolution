# Super-Resolution
Super Resolution using EDSR.
Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR) model trained to convert a Low-Resolution image to a Super-Resolution image. 

# Process (Briefly explanation)

1. We train an EDSR model with low-resolution image as input and the same high-resolution image as output. Generator and Discriminator part is not used here.
2. After training EDSR like above, we fine tune the high resolution (HR) images by building a model using EDSR as generator and a discriminator and train it giving LR and HR images pair as input and taking SR images as output.

# Dataset

DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/

# Model architecture

<br>
<p align="center"><img width="800" src="https://github.com/aryanjain28/Super-Resolution/raw/main/Screenshots/Model_Architecture.png"></p>
<br>


# Screenshots


<br>
<p align="center"><img width="1500" src="https://github.com/aryanjain28/Super-Resolution/raw/main/Screenshots/Screenshot0.png"></p>
<br>

<br>
<p align="center"><img width="1500" src="https://github.com/aryanjain28/Super-Resolution/raw/main/Screenshots/Screenshot1.png"></p>

<p align="center"><img width="1500" src="https://github.com/aryanjain28/Super-Resolution/raw/main/Screenshots/Screenshot2.png"></p>
<br>

<br>
<p align="center"><img width="1500" src="https://github.com/aryanjain28/Super-Resolution/raw/main/Screenshots/Screenshot3.png"></p>
<br>

<br>
<p align="center"><img width="1500" src="https://github.com/aryanjain28/Super-Resolution/raw/main/Screenshots/Screenshot4.png"></p>
<br>


# Things I learnt

1. Training a super resolution model is difficult without a very powerful GPU.
2. About EDSR, WSDR and SR-GAN models.
3. About Perceptual loss and Pixel loss.

# References

1. This project is highly inspired by: http://krasserm.github.io/2019/09/04/super-resolution/#model-training
2. Do check his project for depp explanation: https://github.com/krasserm/super-resolution


Note: I tried training the model on my GPU and failed miserably. Then I tried training it on Colab's GPU, and failed again. At last I had to use pre-trained weights on my model.
