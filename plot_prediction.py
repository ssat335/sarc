#!/usr/bin/env python

"""
Created on Mon Oct 30 10:48:00 2017

<Edit your decription>
"""

__author__      = "Shameer Sathar"
__copyright__   = "Copyright 2017, Shameer Sathar"
__license__     = "MIT"
__version__     = "1.0.1"

from keraspreprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib.image as mpimg
from unet import *
import scipy.misc
def plot_test_data():
    i = 0
    imgs = glob.glob(test_path+"/*.png")
    imgdatas = np.ndarray((1, 512, 512, 1), dtype=np.uint8)
    for imgname in imgs:
        midname = imgname[imgname.rindex("/")+1:]
        img = load_img(test_path + "/" + midname,grayscale = True)
        img = img_to_array(img)
        imgdatas[i] = img
        img_plot = mpimg.imread(test_path + "/" + midname)
        plt.figure()
        imgdatas = imgdatas.astype('float32')
        imgdatas /= 255
        mean = imgdatas.mean(axis = 0)
        imgdatas -= mean
        predicted_data = model.predict(imgdatas, verbose=1)
        #predicted_data[predicted_data > 0.1] = 255
        #predicted_data[predicted_data <= 0.1] = 0
        #scipy.misc.imsave(midname + '_predict.png', predicted_data[0,:,:,0])
        plt.imshow(predicted_data[0, :, :, 0], cmap='hot', interpolation='nearest')#, alpha=0.6)
        plt.imshow(img_plot, alpha=0.8)
    plt.show()

test_path = "./test_nov_2"

myunet = myUnet()

model = myunet.get_unet()

model.load_weights('unet.hdf5')

plot_test_data()
