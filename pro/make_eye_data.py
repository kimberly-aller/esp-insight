import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pdb

def make_eye_data(method):
    '''Makes the eye database for the CNN to train on and to validate
    
    Initially was on RGB but really want this grayscale since my
    training images are actually all grayscale

    '''

    method = 'theano'
    
    eyelists = ['../open_train_nn.list',
                '../open_val_nn.list',
                '../closed_train_nn.list',
                '../closed_val_nn.list']

    #tensorflow form
    outeyefiles = ['../open_train_nn.dat',
                   '../open_val_nn.dat',
                   '../closed_train_nn.dat',
                   '../closed_val_nn.dat']
    #theano form
    outeyefiles = ['../open_train_nn_th.dat',
                   '../open_val_nn_th.dat',
                   '../closed_train_nn_th.dat',
                   '../closed_val_nn_th.dat']

    filenames = [(eyelists[0], outeyefiles[0]),
                 (eyelists[1], outeyefiles[1]),
                 (eyelists[2], outeyefiles[2]),
                 (eyelists[3], outeyefiles[3])]
    xwidth = 24
    ywidth = 24
    nchan = 3

    for (eyelist, outeyefile) in filenames:
        eyefiles = np.genfromtxt(eyelist, dtype=None)
        #pdb.set_trace()
        neyefiles = len(eyefiles)
        if method == 'theano':
            eye_array = np.zeros((neyefiles, nchan, xwidth, ywidth))
        else:
            eye_array = np.zeros((neyefiles, xwidth, ywidth, nchan))
        pos = 0
        for opentraineye in eyefiles:
            #pdb.set_trace()
            img = Image.open('../'+opentraineye)
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
    
    opentrain = np.load(outeyefiles[0]+'.npy')
    closedtrain = np.load(outeyefiles[2]+'.npy')
    xtrain = np.vstack((opentrain,closedtrain))
    xtrain = xtrain/255.0
    ytrain = np.zeros((len(opentrain)+len(closedtrain),1))
    # Classification = 0 for open and 1 for closed    
    ytrain[(len(opentrain)):,0] = 1
    traindat = (xtrain, ytrain)
    pdb.set_trace()
    
    np.save('../openclosed_train_nn_th.dat', xtrain)
    np.save('../openclosed_class_train_nn_th.dat', ytrain)
    
    opentrain = np.load(outeyefiles[1]+'.npy')
    closedtrain = np.load(outeyefiles[3]+'.npy')
    xtrain = np.vstack((opentrain,closedtrain))
    xtrain = xtrain/255.0
    ytrain = np.zeros((len(opentrain)+len(closedtrain),1))
    # Classification = 0 for open and 1 for closed    
    ytrain[(len(opentrain)):,0] = 1
    np.save('../openclosed_val_nn_th.dat', xtrain)
    np.save('../openclosed_class_val_nn_th.dat', ytrain)
