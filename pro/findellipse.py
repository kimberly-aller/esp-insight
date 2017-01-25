import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb

from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

def getell(infile):
    """
    Uses SCIKIT-LEARN to find ellipses in the image

    """
    infile = '/home/kikimei/Insight/esp/haartraining/negative_images/Zorica_Radovic_0001.jpg'
    infile = '/home/kikimei/Insight/esp/closedeyes/Zorica_Radovic_0001_L.jpg'
    
    #pdb.set_trace()
    img = cv2.imread(infile)
    #pdb.set_trace()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img_as_ubyte(gray)
    imgw = (img.shape)[0]
    imgh = (img.shape)[1]

    #img = img_as_ubyte(data.coins()[0:95, 70:370])

    # Detected edges in the image
    edges = canny(img, sigma=3, low_threshold=0.5, high_threshold=0.8)

    fig = plt.figure(1)
    plt.clf()
    ax = fig.gca()
    
    hough_radii = np.arange(5, 20, 2)
    hough_res = hough_circle(edges, hough_radii)
    
    centers = []
    accums = []
    radii = []

    colorimg = color.gray2rgb(img)
    plt.imshow(img)
    plt.show()

    # Get two circles per radius search
    for radius, h in zip(hough_radii, hough_res):
        num_peaks = 2
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:,0], peaks[:,1]])
        pdb.set_trace()
        radii.extend([radius]*num_peaks)
        pdb.set_trace()

        #Plot all circles found
        
        cx, cy = circle_perimeter( (peaks[0])[0], (peaks[0])[1], radius)

        #limit to image range
        cx[cx > imgw] = imgw
        cx[cx < 0] = 0
        cy[cy > imgh] = imgh
        cy[cy < 0] = 0
        
        colorimg[cx, cy] = (220,20,20)
        cx, cy = circle_perimeter( (peaks[1])[0], (peaks[1])[1], radius)
        colorimg[cx, cy] = (220,20,20)
        ax.imshow(colorimg, cmap='gray')
        plt.show()
        
        #pdb.set_trace()

    pdb.set_trace()

    fig2 = plt.figure(2)
    plt.clf()
    colorimg_top = color.gray2rgb(img)
    plt.imshow(colorimg_top, cmap='gray')
    ax2 = fig2.gca()
    
    # Plot top circles
    # ::-1 reverses the order
    # :5 takes 0--5th position = top 5 in order of accums = peak info
    
    for idx in np.argsort(accums)[::-1][:5]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        cx, cy = circle_perimeter(center_x, center_y, radius)
        colorimg_top[cx, cy] = (220,20,20)
            
    ax2.imshow(colorimg_top, cmap='gray')
    plt.show()
