import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
from keras import backend as K
import keras
import findface
import time
from keras.models import load_model
from skimage import color
#import seaborn as sns

# Use the model as a global variable so that we don't need to
# continuously reload it if running a big list of images

eyemodel_file = 'nn_model_final_face_5layers_th.keras'
eyemodel_weights_file = 'nn_model_weights_final_face_5layers_th.keras'
face_model = load_model(eyemodel_file)
#face_model.load_weights(eyemodel_weights_file)

def app(inimage, indir):
    """
    Application wrapper to be put into Flask
    
    INPUT: Image location 

    MODIFICATION:
    2017-01-26 - Created
    2017-01-26 - Fixed bug and mod for flask
    2017-01-30 - Modified for face cutouts
    """

    # PARAMS
    nchannels = 3 # RGB
    modelw = int(100)
    modelh = int(100)

    outimage = indir + '../blinkfinder_face.png'
    
    # Master Eye Open/Closed Detection Model
    eyemodel_file = 'nn_model_final_face_5layers_th.keras'
    eyemodel_weights_file = 'nn_model_weights_final_face_5layers_th.keras'

    tstart = time.time()
    print('reading file')
    #model = load_model(eyemodel_file)
    #model.load_weights(eyemodel_weights_file)
    # Use the global model from the top to decrease load time
    global face_model
    
    tfin = time.time()
    print('Time to load model is %0.4f' %  (tfin-tstart))
    
    # Find your faces in the image
    allfaces = findface.idface(inimage)

    # If nothing is found don't crash
    if allfaces == None:
        print('no faces found')
        return None

    # Read the image file + convert to numpy array + RGB
    treads = time.time()
    img = Image.open(inimage)
    img = img.convert(mode='RGB')
    img = np.asarray(img)
    treadf = time.time()
    print('Time to load image is %0.4f' % (treadf-treads))

    # Checking if cv2 is faster == no same time
    #imgt = cv2.imread(inimage)
    #imgt = cv2.cvtColor(imgt, cv2.COLOR_BGR2RGB)
    #treadf2 = time.time()
    #print('Time to load image with cv2 is %0.4f' % (treadf2-treadf))
    
    #img = 255-img # my gif yalefaces are weird colorschemes
    imgplt = plt.figure(1);
    plt.clf();
    plt.imshow(img, cmap='gray');
    #plt.show()
    ax = imgplt.gca()
    
    # Cutout faces
    blinkarr = []
    blink_prob = []
    finalface_arr = []
    for (fx,fy,fw,fh) in allfaces:
        
        facecut = img[fy:fy+fw,fx:fx+fh]

        # Check if the face is smaller than what the model needs
        if (fw < modelw) | (fh < modelh):
            facecut = cv2.resize(facecut, (modelw, modelh),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            facecut = cv2.resize(facecut, (modelw, modelh),
                                 interpolation=cv2.INTER_CUBIC)

        # Scale the images to the same as the nn model input
        facecut = facecut/255. 

        # change channel location for theano
        face_swap = np.zeros((nchannels, modelw, modelh))
        face_swap[0,:,:] = facecut[:,:,2]
        face_swap[1,:,:] = facecut[:,:,0]
        face_swap[2,:,:] = facecut[:,:,1]
        facecut = face_swap

        # Add initial dimension size=1
        facecut = np.expand_dims(facecut, axis=0)

        tps = time.time()
        prob_closed = face_model.predict(facecut)
        tpf = time.time()
        print('Time to evaluate eyes is %0.4f' % (tpf-tps))
        
        blink_prob.append(prob_closed[0][0])
        #pdb.set_trace()
        if prob_closed > 0.2:
            ax.add_artist(patches.Rectangle((fx,fy),fw,fh,
                                            fill=False,
                                            color='r',
                                            linewidth=3))
            plt.text(fx, fy,'!!!', color='r')
            blinkarr.append(0)
        else:
            blinkarr.append(1)
            ax.add_artist(patches.Rectangle((fx,fy),fw,fh,
                                            fill=False,
                                            color='b',
                                            linewidth=3))
            plt.text(fx, fy,'OK', color='b')

        if len(finalface_arr) == 0:
            finalface_arr = np.array([fx,fy,fw,fh,prob_closed])
        else:
            finalface_arr = np.vstack((finalface_arr,
                                      np.array([fx,fy,fw,fh,prob_closed])))
    print('saving image')
    plt.savefig('app_face_wrap_test.png')

    if 0 in blinkarr:
        textmsg = 'has blinks'
    else:
        textmsg = 'has no blinks!'

    tfin = time.time()
    print('Total time for image is %0.4f' % (tfin-tstart))
    
    #pdb.set_trace()
    #return textmsg
    return finalface_arr


def loop_facelist(inlist):
    """

    Run model on a list of images rather than a single image and return
    output for the full list

    Saves each facecutout image for determining if someone is blinking
    or not. Also will create a txt file with the 1 0 for blinking
    detection to validate the method.

    """

    # Directory output setup
    outdir = '/home/kikimei/Insight/esp/grouppics_cut/'
    outimgdir = '/home/kikimei/Insight/gitdev/esp-insight/'
    outvalidfile = '/home/kikimei/Insight/gitdev/esp-insight/grouppic_valid.list'
    # Read the input list
    imglist = np.genfromtxt(inlist, dtype=None)

    label_list = []
    full_array = []
    
    for inimg in imglist:
        face_data = app(inimg, outimgdir)
        #pdb.set_trace()
        
        # If couldn't find the face then just continue to the next one
        if face_data is None:
            print('No face on %s' % inimg)
            continue
            
        img = Image.open(inimg)
        img = img.convert(mode='RGB')
        img = np.asarray(img)

        cutfig = plt.figure(1)

        # Loop thru each face and display image and ask for user input
        # for whether is blinking or not

        if len(face_data.shape) <= 1:
            fx,fy,fw,fh,fblink = face_data
            facecut = img[fy:fy+fw, fx:fx+fh]
            
            bcolor='b'
            if fblink > 0.2:
                bcolor = 'r'
                
            plt.clf()
            plt.imshow(facecut)
            cutfig.gca().add_artist(patches.Rectangle((fx,fy),fw,fh,
                                    fill=False, color=bcolor, linewidth=3))
            plt.show()
            valid_q = ''
            valid_q = raw_input('Is this person blinking? (y/n):')

            label_list.append(valid_q)

            if valid_q == 'y':
                valid_num = 1
            else:
                valid_num = 0
                
            # Construct the final output array for face positions +
            # blink_probability + actual answer
            full_face = np.array([fx,fy,fw,fh,fblink,valid_num])
            if len(full_array) == 0:
                full_array = full_face
            else:
                full_array = np.vstack((full_array,full_face))

        else:
            for faces in face_data:
                fx,fy,fw,fh,fblink = faces
                facecut = img[fy:fy+fw, fx:fx+fh]
                
                bcolor='b'
                if fblink > 0.2:
                    bcolor = 'r'
                    
                plt.clf()
                plt.imshow(facecut)
                cutfig.gca().add_artist(patches.Rectangle((fx,fy),fw,fh,
                                                          fill=False, color=bcolor, linewidth=3))
                plt.show()
                valid_q = ''
                valid_q = raw_input('Is this person blinking? (y/n):')
                
                label_list.append(valid_q)
                
                if valid_q == 'y':
                    valid_num = 1
                else:
                    valid_num = 0
                    
                # Construct the final output array for face positions +
                # blink_probability + actual answer
                full_face = np.array([fx,fy,fw,fh,fblink,valid_num])
                if len(full_array) == 0:
                    full_array = full_face
                else:
                    full_array = np.vstack((full_array,full_face))

    pdb.set_trace()                               
    np.savetxt(outvalidfile, full_array, delimiter=',', fmt='%10.5f')
    pdb.set_trace()

def plot_eyeprob():

    # Input the results from the fitter (probabilities) + my own
    # classification to determine which threshold to use
    valid_txt = np.genfromtxt('../grouppic_valid_txt.list',
                              dtype=None, delimiter=',')
    valid_nums = np.genfromtxt('../grouppic_valid.list',
                               dtype=None, delimiter=',')
    
    wmaybe = np.where( (valid_txt == 'y (part)') |
                       (valid_txt == 'n (part)'))
    wyes = np.where(valid_txt == 'y')
    wno = np.where(valid_txt == 'n')
    wglasses = np.where('g' in valid_txt)
    wbad = np.where( (valid_txt == 'not person') |
                     (valid_txt == 'shirt face'))

    hfig = plt.figure(1)
    plt.clf()

    # Plot the blinking probability histograms for each class
    blink_prob = valid_nums[:,4]

    # Setup histogram params
    xtitle = 'Blinking Probability'
    ytitle = 'Number'
    bins = 10
    ax = hfig.gca()
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    
    # Plot the histograms to see where thresh should be
    alpha = 0.5
    plt.hist(blink_prob[wyes], color='r',
             bins=bins, alpha=alpha, label='Blinking')
    plt.hist(blink_prob[wno], color='mediumturquoise', bins=bins, alpha=alpha, label='Open')
    plt.hist(blink_prob[wmaybe], color='orchid', bins=bins, alpha=alpha, label='Sleepy')
    plt.legend()
    plt.show()

    # If set thresh to 0.8
    thresh = 0.8

    # Open eyes
    nblink = float(len(wno[0]))
    blinkno = blink_prob[wno]
    nblinkok = len(blinkno[np.where(blinkno >= thresh)])
    print('OpenEyes above thresh: %0.2d%%' % (100*nblinkok/nblink))
    print('OpenEyes below thresh: %0.2d%%' % (100*(nblink-nblinkok)/nblink))

    # Sleepy eyes
    nblink = float(len(wmaybe[0]))
    blinkmaybe = blink_prob[wmaybe]
    nblinkok = len(blinkmaybe[np.where(blinkmaybe >= thresh)])
    print('Sleepy/Blinks above thresh: %0.2d%%' % (100*nblinkok/nblink))
    print('Sleepy/Blinks below thresh: %0.2d%%' % (100*(nblink-nblinkok)/nblink))


    # Blinking Eyes
    nblink = float(len(wyes[0]))
    blinkyes = blink_prob[wyes]
    nblinkok = len(blinkyes[np.where(blinkyes >= thresh)])
    print('Blinking above thresh: %0.2d%%' % (100*nblinkok/nblink))
    print('Blinking below thresh: %0.2d%%' % (100*(nblink-nblinkok)/nblink))


    pdb.set_trace()
