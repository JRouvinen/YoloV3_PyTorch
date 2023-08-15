# import the required libraries
import cv2
import matplotlib.pyplot as plt

path = input("Give path to image folder:")
new_size = (1024, 1024) # new_size=(width, height)

for file in path:
    # read the input image
    img = cv2.imread(file)
    #Get image height width
    h, w, c = img.shape
    # resize the image using different interpolations
    #resize_cubic = cv2.resize(img,(0, 0),fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    resize_area = cv2.resize(img,(0, 0),fx=0.5, fy=0.7, interpolation = cv2.INTER_AREA)
    #resize_linear = cv2.resize(img,(0, 0),fx=2, fy=2, interpolation = cv2.INTER_LINEAR)

    #write resized image
    cv2.imwrite(file,resize_area)

    # display the original and resized images
    plt.subplot(221),plt.imshow(img), plt.title("Original Image")
    #plt.subplot(222), plt.imshow(resize_cubic), plt.title("Interpolation Cubic")
    plt.subplot(223), plt.imshow(resize_area), plt.title("Interpolation Area")
    #plt.subplot(224), plt.imshow(resize_linear), plt.title("Interpolation Linear")
    plt.show()