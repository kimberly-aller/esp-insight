import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
import findeyes
from keras.models import load_model
from skimage import color

def app(inimage):
    """
    Application wrapper to be put into Flask
    
    INPUT: Image location 

    MODIFICATION:
    2017-01-26 - Created
    2017-01-26 - Fixed div/255 error
    """

    # PARAMS
    nchannels = 3 # RGB
    modelw = 24
    modelh = 24
    
    # Master Eye Open/Closed Detection Model
    eyemodel_file = 'nn_model_final.h5'
    model = load_model(eyemodel_file)
    
    # Find your eyes/faces in the image
    alleyes = findeyes.idface(inimage, nofacefind=0)

    # Read the image file + convert to numpy array + RGB
    img = Image.open(inimage)
    img = img.convert(mode='RGB')
    img = np.asarray(img)
    #img = 255-img # my gif yalefaces are weird colorschemes
    imgplt = plt.figure(1)
    plt.clf()
    plt.imshow(img, cmap='gray')
    plt.show()
    ax = imgplt.gca()
    
    # Cutout eye portions for each face
    blinkarr = []
    for (eyex,eyey,eyew,eyeh,facenum) in alleyes:
        
        eyecut = img[eyey:eyey+max([eyew,modelw]),
                     eyex:eyex+max([eyeh,modelh])]

        eyecut = cv2.resize(eyecut, (modelw, modelh))
        eyecut = eyecut/255. # MUST DO THIS BC WE SCALED OUR TEST IMAGES FOR TRAINING!!!
        
        eyecut = np.expand_dims(eyecut, axis=0)

        prob_closed = model.predict(eyecut)

        #pdb.set_trace()
        if prob_closed > 0.2:
            ax.add_artist(patches.Rectangle((eyex,eyey), eyew, eyeh,
                                            fill=False,
                                            color='r',
                                            linewidth=3))
            plt.text(eyex, eyey,'!!!', color='r')
            blinkarr.append(0)
        else:
            blinkarr.append(1)
            plt.text(eyex, eyey,'OK', color='b')
            
    plt.show()
    plt.savefig('temp_app.png')
    #pdb.set_trace()
    
