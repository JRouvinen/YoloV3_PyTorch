#https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
###########################################################################
#Parameters
#----------
#image : ndarray
#    Input image data. Will be converted to float.
#mode : str
#    One of the following strings, selecting the type of noise to add:
#
#    'gauss'     Gaussian-distributed additive noise.
#    'poisson'   Poisson-distributed noise generated from the data.
#    's&p'       Replaces random pixels with 0 or 1.
#    'speckle'   Multiplicative noise using out = image + n*image,where
#                n is uniform noise with specified mean & variance.
############################################################################
import os

import numpy as np
import random
import cv2

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

path = input("Give path to image folder:")
new_size = (1024, 1024) # new_size=(width, height)

def blur(image, size):
   # Averaging
   # You can change the kernel size as you want
   avging = cv2.blur(image, (size, size))
   return avging


img_noise = True
img_blur = True
#Recommended values between 0.01 and 1
noise_min_max = [0.01,0.03, 0.05]
#Recommended values between 2 and 25
blur_min_max = [5,10,15]

for filename in os.listdir(path):
    f = os.path.join(path, filename)
    print(f)
    # read the input image
    img = cv2.imread(f)
    if img_noise is True:
        noise_img = sp_noise(img, 0.05)
        cv2.imwrite('sp_noise.jpg', noise_img)
    if img_blur is True:
        blur_img = blur(img,10)
        cv2.imwrite('blur.jpg', blur_img)


