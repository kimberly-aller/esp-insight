import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb

from skimage import data, color
from skimage.transform import hough_circle
from skimage.transform import hough_ellipse
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.draw import ellipse_perimeter
from skimage.util import img_as_ubyte
from skimage import exposure

def getpupil(infile):
    """
    Uses SCIKIT-LEARN to find eye ellipsese in the image

    """
    
    # Test image setup
    #img_dir = '/home/kikimei/Insight/esp/'
    #infile = img_dir + 'openeyes/Zorica_Radovic_0001_R.jpg'    

    # Read image
    img = cv2.imread(infile)    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_as_ubyte(gray)

    #Add Gaussian Blurr to smooth out features a bit
    # Use threshold based on adaptive mean values
    thresh = cv2.adaptiveThreshold(img, 200, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 5)
    imgblur = cv2.GaussianBlur(img, (3,3), 5)

    # Filter the "median" values i.e. skin tone so that the pupil and
    # eye area are both showing light/dark lines as "dark" rather than
    # being dual values at the extreme high/low
    imgmed = np.median(img)
    imginit = img
    imgfilt = img
    
    imgfilt[img>imgmed] = (imgmed/(255-imgmed))*(-1*img[img>imgmed] + 255)
    imgfilt = imgfilt
    img = imgfilt    

#    img = imgblur
    
    pdb.set_trace()
    
    #Get image params
    imgw = (img.shape)[0]
    imgh = (img.shape)[1]

    # Test image
    #img = img_as_ubyte(data.coins()[0:95, 70:370])

    # Detected edges in the image
    edges = canny(img, sigma=3, low_threshold=0.5, high_threshold=0.8)
    #edges = canny(img, sigma=3, low_threshold=20)
    
    fig = plt.figure(1)
    plt.clf()
    ax = fig.gca()

    # Search for radii starting at 3 pixels and moving to 1/3 the
    # eye-area size in steps of 2 pixels
    hough_radii = np.arange(3, imgw/4, 1)
    hough_res = hough_circle(edges, hough_radii)
    
    centers = []
    accums = []
    radii = []

    colorimg = color.gray2rgb(imginit)
    plt.imshow(img,cmap='gray')
    plt.show()

    #pdb.set_trace()
    # Get two circles per radius search
    for radius, h in zip(hough_radii, hough_res):
        num_peaks = 2
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:,0], peaks[:,1]])
        pdb.set_trace()
        radii.extend([radius]*num_peaks)

        #Plot all circles found

        if len(peaks) != 0:
            cx, cy = circle_perimeter( (peaks[0])[0], (peaks[0])[1], radius)

            #limit to image range
            cx[cx >= imgw] = imgw-1
            cx[cx < 0] = 0
            cy[cy >= imgh] = imgh-1
            cy[cy < 0] = 0
            
            #pdb.set_trace()
            
            colorimg[cx, cy] = (220,20,20)
            cx, cy = circle_perimeter( (peaks[1])[0], (peaks[1])[1], radius)
            cx[cx >= imgw] = imgw-1
            cx[cx < 0] = 0
            cy[cy >= imgh] = imgh-1
            cy[cy < 0] = 0
            
            colorimg[cx, cy] = (220,20,20)
            ax.imshow(colorimg, cmap='gray')
            plt.show()
        
        #pdb.set_trace()

    #pdb.set_trace()

    fig2 = plt.figure(2)
    plt.clf()
    colorimg_top = color.gray2rgb(img)
    plt.imshow(colorimg_top, cmap='gray')
    ax2 = fig2.gca()
    
    # Plot top circles
    # ::-1 reverses the order
    # :5 takes 0--5th position = top 5 in order of accums = peak info

    if len(accums) != 0:
        for idx in np.argsort(accums)[::-1][:1]:
            center_x, center_y = centers[idx]
            radius = radii[idx]
            cx, cy = circle_perimeter(center_x, center_y, radius)
            cx[cx >= imgw] = imgw-1
            cx[cx < 0] = 0
            cy[cy >= imgh] = imgh-1
            cy[cy < 0] = 0
            colorimg_top[cx, cy] = (220,20,20)
        
            ax2.imshow(colorimg_top, cmap='gray')
            plt.show()
        
        # Most strong circle ==> pupil
        ind_top = np.argsort(accums)[::-1][:1]
        pcenter_x, pcenter_y = centers[ind_top]
        pradius = radii[ind_top]
        pheight = accums[ind_top]
        pdb.set_trace()
        return [pcenter_x, pcenter_y, pradius, pheight]
    else:
        return [0, 0, 0, 0]
            
    
def geteye(infile):
    """
    Uses SCIKIT-LEARN to find eye circles in the image

    """

    # -------------------------------------------
    # SETUP PARAMETERS
    
    threshhough = 4
    accuracy = 1
    # angle for orientation away from horizonal < 15deg
    maxangle = 15
    
    # SETUP EYE IMAGE INPUTS
    #img_dir = '/home/kikimei/Insight/esp/'
    #infile = img_dir + 'haartraining/negative_images/Zorica_Radovic_0001.jpg'
    # Read image
    img = cv2.imread(infile)    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img_as_ubyte(gray)
    img = gray
    
    pdb.set_trace()
    #infile = img_dir + 'closedeyes/Zorica_Radovic_0001_L.jpg'
    
    #pdb.set_trace()
    #img = cv2.imread(infile)
    #np2, np98 = np.percentile(img, (2, 98))
    #logimg = exposure.adjust_log(img, 1)
    #conimg = exposure.rescale_intensity(img, in_range=(np2, np98))
    
    #pdb.set_trace()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(img, 200, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 5)
    imgblur = cv2.GaussianBlur(thresh, (3,3), 20)

    # Filter the "median" values i.e. skin tone so that the pupil and
    # eye area are both showing light/dark lines as "dark" rather than
    # being dual values at the extreme high/low
    imgmed = np.median(img)
    imginit = img
    
    imgfilt = img
    
    imgfilt[img>imgmed] = (imgmed/(255-imgmed))*(-1*img[img>imgmed] + 255)

    img = imgfilt    

    plt.imshow(img, cmap='gray')
    plt.show()
    pdb.set_trace()
    
    #vimage = color.rgb2hsv(img)
    #gray = vimage[:,:,2]
    
    pdb.set_trace()
    imgw = (img.shape)[0]
    imgh = (img.shape)[1]

    #pdb.set_trace()

    #This is the test image
    test = 0
    if test == 1:
        img = img_as_ubyte(data.coffee()[0:220, 160:420])
        img = color.rgb2gray(img)
    
    imgw = (img.shape)[0]
    imgh = (img.shape)[1]
    # Detected edges in the image
    edges = canny(img, sigma=2, low_threshold=0.5, high_threshold=0.8)

    #pdb.set_trace()
    hough_radii = np.arange(3, imgw/2, 2)

    pdb.set_trace()
    
    if test == 1:
        hough_res = hough_ellipse(edges, threshold=250,
                                  accuracy=20,
                                  min_size=100, max_size=120)
    else:
        hough_res = hough_ellipse(edges, threshold=threshhough,
                                  accuracy=accuracy, 
                                  min_size=np.min(hough_radii),
                                  max_size=np.max(hough_radii))

    # Sort all ellipses found by order of "strength"
    # require the angle to be within +/- maxangle of horizontal
    # require width > height
    idx = np.where((hough_res['a'] > 0) &
                   (hough_res['b'] > 0) &
                   (hough_res['a'] >= hough_res['b']) &
                   (abs(hough_res['orientation']*180/np.pi - 90) < maxangle))

    hough_res = hough_res[idx]
    hough_res.sort(order='accumulator')
    best_ellipse = list(hough_res[0])
    
    # Make the first plot
    fig2 = plt.figure(2)
    plt.clf()
    colorimg_top = color.gray2rgb(imginit)
    plt.imshow(colorimg_top, cmap='gray')
    ax2 = fig2.gca()

    # Top Ellipse parameters need to be rounded to integers for images
    ellx, elly, ella, ellb = [int(round(par)) for par in best_ellipse[1:5]]
    orien = best_ellipse[5]

    # setup ellipse params
    cx, cy = ellipse_perimeter(ellx, elly, ella, ellb, orien)
    cx[cx >= imgw] = imgw-1
    cx[cx < 0] = 0
    cy[cy >= imgh] = imgh-1
    cy[cy < 0] = 0

    #pdb.set_trace()
    
    colorimg_top[cx,cy] = (220,100,20)

    best_params = np.array([ellx, elly, ella, ellb, orien])

    ax2.imshow(colorimg_top, cmap='gray')
    plt.show()
    
    pdb.set_trace()
    
    for hellipse in hough_res:
        #pdb.set_trace()
        hellipse_l = list(hellipse)
        xc, yc, ac, ab = [int(round(par)) for par in hellipse_l[1:5]]
        orientation = hellipse_l[5]
        cx, cy = ellipse_perimeter(xc, yc, ac, ab, orientation)
        cx[cx >= imgw] = imgw-1
        cx[cx < 0] = 0
        cy[cy >= imgh] = imgh-1
        cy[cy < 0] = 0
        colorimg_top[cx, cy] = (0,200,0)
            
        #ax2.imshow(colorimg_top, cmap='gray')
        #plt.show()
        #pdb.set_trace()

    return best_params
    pdb.set_trace()

def check_pupils(inlist, outfile):
    """
    
    """

    # Get input list of all eyes
    eyefiles = np.genfromtxt(inlist, dtype=None)

    for eyefile in eyefiles:
        pupilres = getpupil(eyefile)
        #eyeres = geteye(eyefile)
        
        if (abs(pupilres[0] - 12) < 3) & (abs(pupilres[0] - 12) < 3):
            pupil_mid = 1
        else:
            pupil_mid = 0

        pupilres = [pupilres[0], pupilres[1], pupilres[2], pupilres[3], pupil_mid]
        if 'pupil_arr' in vars():
            pupil_arr = np.vstack((pupil_arr, np.array([pupilres])))
        else:
            pupil_arr = np.array([pupilres])

        print(pupilres)
        #pdb.set_trace()

    np.savetxt(outfile, pupil_arr, delimiter=',', fmt='%10.5f')

    pdb.set_trace()
    

def check_eyes(inlist, outfile):

    # Get input list of all eyes
    eyefiles = np.genfromtxt(inlist, dtype=None)

    for eyefile in eyefiles:
        #pupilres = getpupil(eyefile)
        pupilres = geteye(eyefile)
        
        if (abs(pupilres[0] - 12) < 3) & (abs(pupilres[0] - 12) < 3):
            pupil_mid = 1
        else:
            pupil_mid = 0

        pupilres = [pupilres[0], pupilres[1], pupilres[2], pupilres[3], pupil_mid]
        if 'pupil_arr' in vars():
            pupil_arr = np.vstack((pupil_arr, np.array([pupilres])))
        else:
            pupil_arr = np.array([pupilres])

        print(pupilres)
        pdb.set_trace()

    np.savetxt(outfile, pupil_arr, delimiter=',', fmt='%10.5f')

    pdb.set_trace()
    
def simple_pupils(inlist, outfile):
    """
    
    """

    # Get input list of all eyes
    eyefiles = np.genfromtxt(inlist, dtype=None)

    for eyefile in eyefiles:

        img = cv2.imread(eyefile)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ysum = np.sum(img, axis=1)
        xsum = np.sum(img, axis=0)

        pupils = getpupil(eyefile)
        
        if (abs(pupilres[0] - 12) < 3) & (abs(pupilres[0] - 12) < 3):
            pupil_mid = 1
        else:
            pupil_mid = 0

        pupilres = [pupilres[0], pupilres[1], pupilres[2], pupilres[3], pupil_mid]
        if 'pupil_arr' in vars():
            pupil_arr = np.vstack((pupil_arr, np.array([pupilres])))
        else:
            pupil_arr = np.array([pupilres])

        print(pupilres)
        #pdb.set_trace()
    
    np.savetxt(outfile, pupil_arr, delimiter=',', fmt='%10.5f')

    pdb.set_trace()
    
