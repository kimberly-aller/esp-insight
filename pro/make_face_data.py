import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pdb

def make_face_data(method):
    '''

    Makes the face database for the CNN to train on and to validate
    for open and closed eyes when run on the full face rather than
    individual eyes.
    
    Initially was on RGB but really want this grayscale since my
    training images are actually all grayscale

    Modification History:
    2017-01-30 - Created

    '''

    method = 'theano'
    
    eyelists = ['../openfaces_list.txt',
                '../openfaces_val_list.txt',
                '../closedfaces_list.txt',
                '../closedfaces_val_list.txt']

    #tensorflow form
    outeyefiles = ['../openface_train_nn.dat',
                   '../openface_val_nn.dat',
                   '../closedface_train_nn.dat',
                   '../closedface_val_nn.dat']
    #theano form
    outeyefiles = ['../openface_train_nn_th.dat',
                   '../openface_val_nn_th.dat',
                   '../closedface_train_nn_th.dat',
                   '../closedface_val_nn_th.dat']

    outeyecombo = ['../openclosed_face_train_nn_th.dat',
                   '../openclosed_face_train_class_nn_th.dat',
                   '../openclosed_face_val_nn_th.dat',
                   '../openclosed_face_val_class_nn_th.dat']

    filenames = [(eyelists[0], outeyefiles[0]),
                 (eyelists[1], outeyefiles[1]),
                 (eyelists[2], outeyefiles[2]),
                 (eyelists[3], outeyefiles[3])]
    xwidth = 100
    ywidth = 100
    nchan = 3

    # Combine each face array into one by reading in each image from
    # the list and altering the format
    for (eyelist, outeyefile) in filenames:
        
        print(eyelist)
        pdb.set_trace()
        eyefiles = np.genfromtxt(eyelist, dtype=None)
        
        neyefiles = len(eyefiles)
        if method == 'theano':
            eye_array = np.zeros((neyefiles, nchan, xwidth, ywidth))
        else:
            eye_array = np.zeros((neyefiles, xwidth, ywidth, nchan))
        pos = 0
        for opentraineye in eyefiles:
            #pdb.set_trace()
            img = Image.open(opentraineye)
            img = img.convert(mode='RGB')
            img = np.asarray(img)
            if method == 'theano':
                img_swap = eye_array[pos]
                img_swap[0,:,:] = img[:,:,2]
                img_swap[1,:,:] = img[:,:,0]
                img_swap[2,:,:] = img[:,:,1]
                #pdb.set_trace()
                eye_array[pos] = img_swap
                
            else:
                eye_array[pos] = img
            #pdb.set_trace()
            pos += 1
        np.save(outeyefile, eye_array)
        pdb.set_trace()
        print(eyelist)
    # Load in the open/closed face arrays to combine + scale
    opentrain = np.load(outeyefiles[0]+'.npy')
    closedtrain = np.load(outeyefiles[2]+'.npy')
    xtrain = np.vstack((opentrain,closedtrain))
    xtrain = xtrain/255.0
    ytrain = np.zeros((len(opentrain)+len(closedtrain),1))
    # Classification = 0 for open and 1 for closed    
    ytrain[(len(opentrain)):,0] = 1
    traindat = (xtrain, ytrain)
    pdb.set_trace()

    # Output the open+closed face arrays into one output for Keras
    #np.save('../openclosed_train_nn_th.dat', xtrain)
    #np.save('../openclosed_class_train_nn_th.dat', ytrain)
    np.save(outeyecombo[0], xtrain)
    np.save(outeyecombo[1], ytrain)

    # Load in the open/closed face arrays for *val* to combine + scale
    opentrain = np.load(outeyefiles[1]+'.npy')
    closedtrain = np.load(outeyefiles[3]+'.npy')
    xtrain = np.vstack((opentrain,closedtrain))
    xtrain = xtrain/255.0
    ytrain = np.zeros((len(opentrain)+len(closedtrain),1))
    # Classification = 0 for open and 1 for closed    
    ytrain[(len(opentrain)):,0] = 1

    # Output the open+closed face arrays into one output for validation 
    #np.save('../openclosed_val_nn_th.dat', xtrain)
    #np.save('../openclosed_class_val_nn_th.dat', ytrain)
    np.save(outeyecombo[2], xtrain)
    np.save(outeyecombo[3], ytrain)
