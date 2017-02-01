import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import pdb
from PIL import Image
import time
def idface(imgname):

    # OUTPUTS/INPUTS 
    imgname_spl = imgname.split('.')
    imgname_ed = imgname_spl[0] + '-ed.' + imgname_spl[1]

    # SETUP FACE parameters
    faceSF = 1.1 # face ScaleFactor
    faceMN = 3 # face MinNeighbors
        
    # Getting the Haar-cascade from OpenCV
    haarface = '/home/kikimei/Insight/esp/opencv/data/haarcascades/' + \
    'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haarface)

    if ('.jpg' in imgname) | ('.png' in imgname):
        img0 = cv2.imread(imgname)
    else:
        img0 = Image.open(imgname)
        img0 = img0.convert(mode='RGB')
        img0 = np.asarray(img0)

    imgw = (img0.shape)[0]
    imgh = (img0.shape)[1]

    print(imgw,imgh)

    # assume 4x6 standard photo - need to shrink to 600x400 for opencv?
    if imgh > imgw:
        smw = 200
        smw = 400 # need at least 400 to find faces for camera phone image b/c lower res than dslr
        smh = int((float(imgh)/imgw)*smw)

        #smh = 450
    else:
        smh = 200
        smh = 400
        smw = int((float(imgw)/imgh)*smh)
        #smw = 450
        

    #pdb.set_trace()
    #Resize images (resizing/etc is HEIGHT,WIDTH ><)
    #  -- only resize if larger
    if ((img0.shape)[0] > smw) & ((img0.shape)[1] > smh):    
        img = cv2.resize(img0,(smh, smw), interpolation=cv2.INTER_CUBIC)
        factorsc = float(imgw)/smw
    else:
        img = img0
        factorsc = float(1)

    print(smw, smh)
    
    #Image Scale Factor
    xrange0 = (img0.shape)[1]
    yrange0 = (img0.shape)[0]
    xrangesm = (img.shape)[1]
    yrangesm = (img.shape)[0]
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    
    # Find Faces in image: (image, scalefactor, nearestneighbors)
    stime = time.time()
    faces = face_cascade.detectMultiScale(gray, faceSF, faceMN)
    #pdb.set_trace()

    # Plot the full image:
    full_fig = plt.figure(1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # Faces locations on full-size image (unscale dimensions)
    if len(faces) < 1:
        print('did not find face')
        return None

    faces_big = faces*factorsc

    for (fx,fy,fw,fh) in faces_big:
        full_fig.gca().add_artist(patches.Rectangle((fx,fy),fw,fh,
                                                    fill=False,
                                                    color='r',
                                                    linewidth=3))
    ftime = time.time()
    time_tot = (ftime-stime)
    print('Total time to find faces: %0.4f' % time_tot)
    #pdb.set_trace()
               
    return faces_big
