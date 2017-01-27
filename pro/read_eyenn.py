from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import pdb
from PIL import Image
import numpy as np

def eyemodel(modelname):


    model = load_weights(modelname)
