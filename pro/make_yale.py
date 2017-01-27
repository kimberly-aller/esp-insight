import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
from PIL import Image

from skimage import data, color
from skimage.transform import hough_circle
from skimage.transform import hough_ellipse
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.draw import ellipse_perimeter
from skimage.util import img_as_ubyte
from skimage import exposure

import matplotlib.patches as patches
import math

import findeyes

def make_yale(infile, infileclosed):

    # hapy_eyes.list
    # sleepy_eyes.list
    
    dirname = '/home/kikimei/Insight/esp/yalefaces/'
    
    # Outfile is mod of infile
    imglist = np.genfromtxt(dirname+infile, dtype=None)
    imglistclosed = np.genfromtxt(dirname+infileclosed, dtype=None)
    
    for (imgin, imgingclose) in (imglist, imglistclosed):

        img = Image.open(dirname+imgin)
        img = img.convert(mode='RGB')        
        img = np.asarray(img)
        imgclosed = Image.open(dirname+imginclose)
        eye_params = findeyes.idface(dirname+imgin, 0)

        imgname_out = dirname+imgin + '-eyes.'

        ind = 0
        for eyes in eye_params:
            
            eyecut = img[(eyes[1]):(eyes[1]+eyes[3]),
                         (eyes[0]):(eyes[0]+eyes[2]),:]
            plt.imshow(eyecut)
            plt.show()
            eyecut_f = Image.fromarray(eyecut, 'RGB')
            if ind == 0:
                eyecut_f.save(imgname_out+'R.jpg')
            elif ind == 1:
                eyecut_f.save(imgname_out+'L.jpg')
            else:
                print('you have three eyes')

            ind += 1

    
