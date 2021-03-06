from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import pdb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def eyemodel(modelname):

    # SETUP METHOD
    method = 'theano'
    
    #Load model from file
    if method == 'theano':
        inmodel = 'nn_model_final_th.keras'
        inxvalid = '../openclosed_val_nn_th.dat.npy'
        inyvalid = '../openclosed_class_val_nn_th.dat.npy'
        outpng = 'OpenClosedEyes_nn_hist_th.png'
    else:
        inmodel = 'nn_model_final_tf.keras'
        inxvalid = '../openclosed_val_nn.dat.npy'
        inyvalid = '../openclosed_class_val_nn.dat.npy'
        outpng = 'OpenClosedEyes_nn_hist.png'
        
    model = load_model(inmodel)
    pdb.set_trace()
        
    print('loaded eye model')

    #pdb.set_trace()
    # Get Validation Data
    xvalid = np.load(inxvalid)
    yvalid = np.load(inyvalid)
    yvalid = yvalid[:,0]
    closed_ll = np.zeros(len(yvalid))

    pos = 0
    for eyetest in xvalid:
        eyeshape = eyetest.shape
        #pdb.set_trace()
        eye = np.zeros((1, eyeshape[0], eyeshape[1], eyeshape[2]))
        eye[0] = eyetest
        prob_closed = model.predict(eye)
        closed_ll[pos] = prob_closed[0][0]
        pos += 1
        #pdb.set_trace()

    wclosed = np.where(yvalid == 1)
    wopen = np.where(yvalid == 0)

    outhist = plt.figure(1)
    plt.clf()
    ax = outhist.gca()

    # Setting threshold to remove too many false positives/negatives
    thresh = 0.2
    closedeyes = closed_ll[wclosed]
    openeyes = closed_ll[wopen]

    nclosedbelow = len(closedeyes[closedeyes<=thresh])
    nopenabove = len(openeyes[openeyes>=thresh])

    # False positives --> actually are open eyes
    nfp = 100*float(nopenabove)/len(openeyes)
    # False negatives --> actually are closed eyes
    nfn = 100*float(nclosedbelow)/len(closedeyes)

    print('FP of %d%%' % nfp)
    print('FN of %d%%' % nfn)
    
    outhist.gca().set_xlabel('Closed Eye Likelihood')
    outhist.gca().set_ylabel('Number')
    plt.hist(closed_ll[wclosed], alpha=0.5, label='Closed Eyes')
    plt.hist(closed_ll[wopen], alpha=0.5, label='Open Eyes')
    plt.plot((thresh,thresh),(0,450), 'k-', color='crimson', linewidth=3)
    plt.legend()
    plt.show()
    plt.savefig(outpng)
    pdb.set_trace()
