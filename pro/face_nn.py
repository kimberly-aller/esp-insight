# Using Convolutional Neural-Networks to classify open/closed eyes
#using the training images
#
# METHOD:
#
#  Train the keras convolutional neural network to classify faces with
#  open eyes and closed eyes.
#
# RESULTS:
#
#  93% with 12%FP, 4% FN - 5 layers + 25 epochs   
#
# HISTORY: 2017-01-30 - Modified from the eyes nn
#
#

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import pdb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

tstart = time.time()

np.random.seed(127)

#Setup image parameters
img_width = 100
img_height = 100
batch_size = 32
nchannels = 3
method = 'theano'

#Setup directories
nb_train_samples = 1970
nb_val_samples = 453
nb_epoch = 25

# Get Training Data +  Validation Data
if method == 'theano':
    xtrain = np.load('../openclosed_face_train_nn_th.dat.npy')
    ytrain = np.load('../openclosed_face_train_class_nn_th.dat.npy')
    xvalid = np.load('../openclosed_face_val_nn_th.dat.npy')
    yvalid = np.load('../openclosed_face_val_class_nn_th.dat.npy')
    outmodel = 'nn_model_final_face_5layers_th.keras'
    outweights = 'nn_model_weights_final_face_5layers_th.keras'
    outjson = 'nn_model_final_face_5layers_th.json'
else:
    xtrain = np.load('../openclosed_face_train_nn.dat.npy')
    ytrain = np.load('../openclosed_face_class_train_nn.dat.npy')
    xvalid = np.load('../openclosed_face_val_nn.dat.npy')
    yvalid = np.load('../openclosed_face_class_val_nn.dat.npy')
    outmodel = 'nn_model_final_face_tf.keras'
    
#rotation_range=40,
#width_shift_range=0.2,
#height_shift_range=0.2,

# add inception layer == border_mode='same' - prevent from losing edges
# + more convolutions to get down to 1

model = Sequential()

if method == 'theano':    
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(nchannels,img_width, img_height)))
else:
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_width, img_height,nchannels)))

model.add(Activation('relu'))
# 100 --> 50
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

# 50 --> 25
model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

#25 -- 12 (4 layers)
model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

# 12 -- 6 (5 layers)
model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# for binary things 0-1 
model.add(Activation('sigmoid'))
#model.add(Activation('softmax'))

#pdb.set_trace()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(xtrain, ytrain,
          validation_split=0.2,
          batch_size=batch_size,
          nb_epoch=nb_epoch)

#Output model
model.save(outmodel)

model.save_weights(outweights)

model_json = model.to_json()
with open(outjson, "w") as json_file:
    json_file.write(model_json)
    
score = model.evaluate(xvalid, yvalid)

print('accuracy: %3d%%' % float(score[1]*100))

if method == 'theano':
    test = np.zeros((1,nchannels,img_width,img_height))
else:
    test = np.zeros((1,img_width, img_height, nchannels))
    
test[0] = xvalid[0]

model.predict(test)

tfin = time.time()

print('Total time: %5d%%' % tfin-tstart)
pdb.set_trace()

