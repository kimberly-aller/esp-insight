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

    json_file = open('nn_model_final.json', 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    #Load model weights from file
    model = load_model('nn_model_final.h5')
    print('loaded eye model')

    #pdb.set_trace()
    # Get Validation Data
    xvalid = np.load('../openclosed_val_nn.dat.npy')
    yvalid = np.load('../openclosed_class_val_nn.dat.npy')
    yvalid = yvalid[:,0]
    closed_ll = np.zeros(len(yvalid))

    pos = 0
    for eyetest in xvalid:
        eyeshape = eyetest.shape
        eye = np.zeros((1, eyeshape[0], eyeshape[1], eyeshape[2]))
        eye[0] = eyetest
        prob_closed = model.predict(eye)
        closed_ll[pos] = prob_closed[0][0]
        pos += 1
        #pdb.set_trace()

    wclosed = np.where(yvalid == 0)
    wopen = np.where(yvalid == 1)

    outhist = plt.figure(1)
    plt.clf()
    ax = outhist.gca()
    outhist.gca().set_xlabel('Open Eye Likelihood')
    outhist.gca().set_ylabel('Number')
    plt.hist(closed_ll[wclosed], alpha=0.5, label='Closed Eyes')
    plt.hist(closed_ll[wopen], alpha=0.5, label='Open Eyes')
    plt.legend()
    plt.show()
    plt.savefig('OpenClosedEyes_nn_hist.png')
    pdb.set_trace()
