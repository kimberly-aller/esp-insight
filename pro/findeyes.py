import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def idface(imgname, nofacefind):
    """
    Finding the face using OpenCV
    """

    plt.clf()
    # OUTPUTS/INPUTS 
    imgname_spl = imgname.split('.')
    imgname_ed = imgname_spl[0] + '-ed.' + imgname_spl[1]

    # SETUP FACE parameters
    faceSF = 1.1 # face ScaleFactor
    faceMN = 3 # face MinNeighbors
    
    feye = 0.5 # eye in top half of face
    feyetrim = 0.05 # eye not in top/bottom 5% of the face
    fover = 0.1 # eye overlap within 10%
    fwover = 0.3 # widths overlapping within 30% size

    eyeSF = 1.01 # eye ScaleFactor
    eyeMN = 2 # eye MinNeighbors
    
    
    # Getting the Haar-cascade from OpenCV
    haarface = '/home/kikimei/Insight/esp/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
    haareye = '/home/kikimei/Insight/esp/opencv/data/haarcascades/haarcascade_eye.xml'
    haareyel = '/home/kikimei/Insight/esp/opencv/data/haarcascades/haarcascade_lefteye_2splits.xml'
    haareyer = '/home/kikimei/Insight/esp/opencv/data/haarcascades/haarcascade_righteye_2splits.xml'
    
    face_cascade = cv2.CascadeClassifier(haarface)
    eye_cascade = cv2.CascadeClassifier(haareye)
    leye_cascade = cv2.CascadeClassifier(haareyel)
    reye_cascade = cv2.CascadeClassifier(haareyer)

    img0 = cv2.imread(imgname)
    imgw = (img0.shape)[0]
    imgh = (img0.shape)[1]

    print(imgw,imgh)

    # assume 4x6 standard photo - need to shrink to 600x400 for opencv?
    if imgh > imgw:
        smw = 200
        smh = 300
    else:
        smw = 300
        smh = 200

    #Resize images (resizing/etc is HEIGHT,WIDTH ><)
    #  -- only resize if larger
    if ((img0.shape)[0] > smw) & ((img0.shape)[1] > smh):    
        img = cv2.resize(img0,(smh, smw), interpolation=cv2.INTER_CUBIC)
        factorsc = float(imgw)/smw
    else:
        img = img0
        factorsc = float(1)
        
    #Image Scale Factor
    xrange0 = (img0.shape)[1]
    yrange0 = (img0.shape)[0]
    xrangesm = (img.shape)[1]
    yrangesm = (img.shape)[0]
    

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    
    # Find Faces in image: (image, scalefactor, nearestneighbors)
    faces = face_cascade.detectMultiScale(gray, faceSF, faceMN)
    #pdb.set_trace()
    
    # Faces locations on full-size image (unscale dimensions)
    if len(faces) < 1:
        print('did not find face')
        if nofacefind == 0:
            return np.array(-1)
        else:
            faces = np.array([[0,0, min(gray.shape), min(gray.shape)]])            

    faces_big = faces*factorsc
    
    # run thru the x,y,w,h coords
    eyes_arr = []
    eyes_arr_big = []
    
    for (x,y,w,h) in faces_big:
        #search within each "face"
        x2 = int(math.ceil(x+w))
        y2 = int(math.ceil(y+h))
        x1 = int(round(x))
        y1 = int(round(y))
        w = int(round(w))
        h = w

        # Eyes can't be larger than 50% of face or less than 10%
        mineye = int(round(h*0.1))
        maxeye = int(round(h*0.5))
        
        #pdb.set_trace()
        #cv2.rectangle(img0,(x1,y1),(x2,y2),(255,0,0),4)

        # Make the plot and show the rectangles around faces
        fig1 = plt.figure(1)
        plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        fig1.gca().add_artist(patches.Rectangle((x1,y1),w,h, fill=False,
                                          color='r', linewidth=3))
        plt.show()
        
        #cv2.namedWindow('facetest',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('facetest', 800,800)
        #cv2.imshow('facetest',img0)
        
        #crop the face location
        roi_gray = gray0[y1:y1+h, x1:x1+w]
        roi_color = img0[y1:y1+h, x1:x1+w]
        
        #Find eyes within the cutout face
        #Scale factor to detect the eyes with smaller sizes
#        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.01,
#                                            minNeighbors=2, minSize=(mineye, mineye),
#                                            maxSize=(maxeye,maxeye))
        reyes = reye_cascade.detectMultiScale(roi_gray, scaleFactor=eyeSF,
                                             minNeighbors=eyeMN, minSize=(mineye, mineye),
                                             maxSize=(maxeye,maxeye))
        leyes = leye_cascade.detectMultiScale(roi_gray, scaleFactor=eyeSF,
                                              minNeighbors=eyeMN, minSize=(mineye, mineye),
                                              maxSize=(maxeye,maxeye))
        # Allow for non-detection of eyes
        if (len(reyes) < 1):
            #pdb.set_trace()
            print('missing right eyes')
            idr = []
        if (len(leyes) < 1):
            #pdb.set_trace()
            print('missing left eyes')
            idl = []
        if (len(reyes) >= 1):
            idr = np.where((reyes[:,1] < (roi_gray.shape)[1]*feye) &
                           (reyes[:,1] > (roi_gray.shape)[1]*feyetrim))
        if (len(leyes) >= 1):
            idl = np.where((leyes[:,1] < (roi_gray.shape)[1]*feye) &
                           (leyes[:,1] > (roi_gray.shape)[1]*feyetrim))

        #pdb.set_trace()
        
        if (len(idl) != 0) & (len(idr) != 0):
            eyes = np.vstack((reyes[idr],leyes[idl]))
        elif (len(idl) != 0) & (len(idr) == 0):
            eyes = leyes[idl]
        elif (len(idr) != 0) & (len(idl) == 0):
            eyes = reyes[idr]
        else:
            return None
        
        #pdb.set_trace()
        
        # Assume eyes are in the TOP half of the face
        eyes0 = eyes
        eyesy = eyes[:,1]
        idx = np.where((eyesy < (roi_gray.shape)[1]*feye) &
                       (eyesy > (roi_gray.shape)[1]*feyetrim))
        eyes = eyes[idx]
        
        #PLot each face with overplotted-eyes
        eyefig = plt.figure(2)
        plt.clf()
        plt.imshow(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
        plt.show()
        
        #pdb.set_trace()
        
        # check if find both eyes
        if len(eyes) >= 1:
            if eyes.size < 8:
                print('missing one eye')
                emin = int(np.round(0.75*(eyes[0])[2]))
                eyes2 = eye_cascade.detectMultiScale(roi_gray, scaleFactor=eyeSF,
                                                     minSize=(emin,emin), minNeighbors=eyeMN-1)
                if len(eyes2) > 1:
                    print('found second eye')
                    #pdb.set_trace()
                    #idx = np.where(eyes[:,1] < (roi_gray.shape)[1]*feye)
                    #eyes = eyes[idx]
                #pdb.set_trace()
        else:
            print('missing eyes (in unallowed regions')
            #pdb.set_trace()
            return None

        # Obviously we do not have 3+ eyes
        if len(eyes) > 2:
            #pdb.set_trace()
            for (ex,ey,ew,eh) in eyes:
                try:
                    eyedel
                except NameError:
                    pass
                else:
                    if np.array([ex,ey,ew,eh]) in eyedel:
                        #pdb.set_trace()
                        continue
                distx = abs(float(ex)-eyes[:,0])
                distx2 = abs(float(ex+ew)-(eyes[:,0]+eyes[:,2]))
                disty = abs(float(ey)-eyes[:,1])
                disty2 = abs(float(ey+ew)-(eyes[:,1]+eyes[:,2]))
                distw = abs(float(ew)-eyes[:,2])
                wnear = np.where( ((distx+disty != 0) & (distx2+disty2 != 0))
                                  & ( ((distx < (w*fover)) & (disty < (w*fover)))
                                      | ((distx2 < (w*fover)) & (disty2 < (w*fover))))
                                  & ((distw/ew) < fwover))
                #pdb.set_trace()
                #If the list is not empty
                if wnear:
                    try:
                        eyedel
                    except NameError:
                        eyedel = eyes[wnear]
                    else:
                        eyedel = np.vstack((eyedel, eyes[wnear]))
                    eyes = np.delete(eyes, wnear, 0)

                else:
                    try:
                        eyesf
                    except NameError:
                        eyesf = np.array([ex,ey,ew,eh])
                    else:
                        eyesf = np.vstack((eyesf, np.array([ex,ey,ew,eh])))
                #pdb.set_trace()

        # Construct final eyes array with coordinates for the full image too
        #eyes_arr = [eyes_arr, eyes]

        #pdb.set_trace()
            
        for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyefig.gca().add_artist(patches.Rectangle((ex, ey), ew, eh,
                                                      fill=False, color='r',
                                                      linewidth=3))
            #pdb.set_trace()

            #Check if the array already exists
            if len(eyes_arr) > 0:
                eyes_arr = np.vstack((eyes_arr,np.array([ex,ey,ew,eh])))
                eyes_arr_big = np.vstack((eyes_arr_big, np.array([x1+ex, y1+ey, ew, eh])))
            else:
                eyes_arr = np.array([[ex,eh,ew,eh]])
                eyes_arr_big = np.array([[x1+ex, y1+ey, ew, eh]])

        plt.show()
        #pdb.set_trace()
        #plt.clf()

    # After finding all eyes/faces
#    cv2.namedWindow('id-ed',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('id-ed', 800,800)
#    cv2.imshow('id-ed', img0)
#    cv2.imwrite(imgname_ed, img0)

    #CV2 is BGR not RGB

    full_fig = plt.figure(1)
    plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    plt.show()

    #pdb.set_trace()
    
    for (ex,ey,ew,eh) in eyes_arr_big:
        full_fig.gca().add_artist(patches.Rectangle((ex,ey),ew,eh,fill=False,
                                                    color='r', linewidth=3))
    plt.show()
    return eyes_arr_big
    #pdb.set_trace()

def CheckEyesLoc(eyelocfile, eyedir, outfile):
    """
    
    """

    # SETUP PArams
    fleft = 0.5 # left eyes on left half of image
    
    # Read the eyelocation file
    eyefile = np.genfromtxt(eyelocfile, dtype=None)
    eyefile = eyefile
    pdb.set_trace()

    fdir = str.split(eyelocfile, '/')
    fdir = fdir[:-1]
    dirs = ''
    for dirpart in fdir:
        dirs = dirs+dirpart + ('/')

    # All eye files are in the same directory
    dirs = dirs + eyedir + '/'

    #neyes_arr = np.zeros((2, len(eyefile)))
    neyes_arr = []
    facefail = [] # if no "face" detected b/c of too zoomed-in
    for (efilename,ex1,ey1,ex2,ey2) in eyefile:
        fimg_i = dirs + '/' + efilename
        img_i = cv2.imread(fimg_i)        
        
        #pdb.set_trace()
        eyeloc = idface(fimg_i, nofacefind=1)
        #pdb.set_trace()

        # No eyes detected
        if eyeloc == None:
            nright = 0
            nleft = 0
            nface = 1
        elif len(eyeloc[eyeloc < 0]) == 1:
            #pdb.set_trace()
            nright = 0
            nleft = 0
            nface = 0
        else:
            # Find number of left/right eyes
            wleft = np.where(eyeloc[:,0] < (img_i.shape)[0]*fleft)
            nright = len(eyeloc) - len(wleft) 
            nleft = len(wleft)
            nface = 1
            
        #pdb.set_trace()
        if len(neyes_arr) > 0:
            neyes_arr = np.vstack((neyes_arr,np.array([nleft, nright])))
            neyes_arr2 = np.vstack((neyes_arr2,np.array([nleft, nright, nface])))
            facefail = np.vstack((facefail,np.array([nface])))
        else:
            neyes_arr = np.array([nleft, nright])
            neyes_arr2 = np.array([nleft, nright, nface])
            facefail = np.array([nface])
    
    #pdb.set_trace()
    histfig = plt.figure(1)
    plt.clf()
    plt.hist(neyes_arr[:,0])
    plt.ylabel = '# Eyes Detected'
    plt.show()
    histfig2 = plt.figure(2)
    plt.clf()
    ax = histfig2.add_subplot(111, aspect='equal')
    nleyes = len(neyes_arr[np.where(neyes_arr[:,0] != 0)])
    nreyes = len(neyes_arr[np.where(neyes_arr[:,1] != 0)])
    plt.plot([0,2],[0,1],color='white')

    #harr = [[0,float(nleyes)/len(eyefile)],[1,float(nreyes)/len(eyefile)]]
    histfig2.gca().add_artist(patches.Rectangle((0,0),1,float(nleyes)/len(eyefile)))
    histfig2.gca().add_artist(patches.Rectangle((1,0),1,float(nreyes)/len(eyefile)))
    #ax.add_patch(patches.Rectangle((1,0),1,float(nreyes)/len(eyefile), facecolor='red'))
    #ax.add_patch(patches.Rectangle((1,0),1,float(nleyes)/len(eyefile)))
    plt.ylabel = '% Eyes Detected'
    #ax.set_xticklabels = ['Left', 'Right']
    #ax.set_xticks = [0, 1, 2]
    #plt.hist(harr)
    plt.show()

    np.savetxt(outfile, neyes_arr2, delimiter=',', fmt='%10.5f')
    pdb.set_trace()
    
def plot_check(resfile, outpng):
    """
    Plotting the results from the open/closed eye database checks to
    see how well the program can even detect where any eyes are
    whether open or closed so that I can know if I can get the
    relevant area for cropping to do the final decision of closed
    vs open.
    
    Modification History:
    2017-01-23 - Kimberly
    """
    
    # Read input file
    res = np.genfromtxt(resfile, dtype=None, delimiter=',')
    
    # How many faces weren't findable using the code b/c it searches for "faces" first
    nbadface = len(res[res[:,2] == 0])
    nfaces = len(res)
    fper = 100*float(nfaces-nbadface)/nfaces
    
    print('Number of faces that we could not use for eye ' +
          'searching b/c cropped too small is: %d' % nbadface)
    print('Total number of faces was: %d' % nfaces)
    
    # Plot the percentages using only the faces we could find
    
    wfaceok = np.where(res[:,2] != 0)
    neyes_fin = res[wfaceok]
    nfaces_fin = len(neyes_fin)
    
    # Number of left/right eyes detected
    nleyes = len(neyes_fin[np.where(neyes_fin[:,0] != 0)])
    nreyes = len(neyes_fin[np.where(neyes_fin[:,1] != 0)])
    
    # Add histogram
    
    histfig = plt.figure(1)
    plt.clf()
    plt.show()
    plt.ylabel = '% Eyes Detected'
    ax = histfig.add_subplot(111) #, aspect='equal')
    plt.xlim(0,2)
    plt.ylim(0,100)
    histfig.gca().set_xticks([0.5,1.5])
    histfig.gca().set_xticklabels(['Left Eyes', 'Right Eyes'])
    histfig.gca().set_ylabel('% Eyes Detected')
    histfig.gca().set_title('Closed Eyes')
    histfig.gca().add_artist(patches.Rectangle((0.1,0),0.8,100*float(nleyes)/nfaces_fin))
    histfig.gca().add_artist(patches.Rectangle((1.1,0),0.8,100*float(nreyes)/nfaces_fin))
    
    plt.show()
    plt.savefig(outpng)
    lper = 100*float(nleyes)/nfaces_fin
    rper = 100*float(nreyes)/nfaces_fin
    print('Detected %d %% of Left Eyes' % lper)
    print('Detected %d %% of Right Eyes' % rper)
    print('Detected %d%% Faces' % fper)
    pdb.set_trace()
