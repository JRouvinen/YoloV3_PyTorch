###########################################################################
# This component creates black and white, noise and blur versions of given images.
#
#
#
#
############################################################################

import os
import shutil
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

def blur(image, size):
   # Averaging
   # You can change the kernel size as you want
   avging = cv2.blur(image, (size, size))
   return avging

def black_white(image):
    #originalImage = cv2.imread('C:/Users/N/Desktop/Test.jpg')
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    return grayImage

#How many file will be edited from original file count (in %)
percent_of_files = 20
number_of_files = 0
#Noise settings
img_noise = True
#Recommended values between 0.01 and 1
noise_val = 0.05
#Blur settings
img_blur = True
#Recommended values between 2 and 25
blur_val = 10
#Black and white settings
black_and_white = True
#Image size settings
new_size = (1024, 1024) # new_size=(width, height) -> not used in this version
resize = True
#Version number
ver = 0.1
#Files to process settigs
random_start_point = random.randint(0, 10)

path = input("Give path to image folder:")

#Print header
print(f"###### Image distortion generator - {ver} ######")
#Count number of files to process
print("----- Counting number of files to process -----")
for filename in os.listdir(path):
    number_of_files += 1
print(f"----- {number_of_files} files found -----")

step = int(number_of_files*(percent_of_files/100))/2
#print(f"step {step}")
#print(f"random start {random_start_point}")
file_indx = 0
step_count = 0

#Create noise and blur files
for filename in os.listdir(path):
    if file_indx > random_start_point:
        if step_count == step:
            comb_rand = random.randint(1, 7)
            # 1: b&w , 2: noise ,3: blur , 4: b&w+noise, 5: noise+blur, 6: b&w+blur, 7: b&w+noise+blur
            f = os.path.join(path, filename)
            # read the input image
            img = cv2.imread(f)
            new_name_add = "_"
            img_modified = False
            if black_and_white is True:
                if comb_rand == 1 or comb_rand == 4 or comb_rand == 6 or comb_rand == 7:
                    black_white_img = black_white(img)
                    img = black_white_img
                    new_name_add += "bw"
                    img_modified = True

            if img_noise is True:
                if comb_rand == 2 or comb_rand == 4 or comb_rand == 5 or comb_rand == 7:
                    #Create new image with noise
                    noise_img = sp_noise(img, noise_val)
                    img = noise_img
                    new_name_add += "_noise"
                    img_modified = True

            if img_blur is True:
                if comb_rand == 3 or comb_rand >= 5:
                    blur_img = blur(img,blur_val)
                    img = blur_img
                    new_name_add += "_blur"
                    img_modified = True

            if img_modified is True:
                new_file_name = filename.split(".")
                new_file_name = str(new_file_name[0]) + new_name_add + "." + str(new_file_name[1])
                new_file_path = f = os.path.join(path, new_file_name)
                cv2.imwrite(new_file_path, img)
                # Get original image labels and create new for noise image
                label_path = path.replace("images", "labels\\") + filename.replace("jpg", "txt")
                new_label_path = label_path.replace(".txt", new_name_add+".txt")
                shutil.copyfile(label_path, new_label_path)
            step_count = 0
        else:
            step_count += 1

    else:
        file_indx += 1
