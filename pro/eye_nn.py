# Using Convolutional Neural-Networks to classify open/closed eyes
#using the training images
#
# METHOD:
#
#
# RESULTS:
#
#   accuracy = 50 % ish still
#
# HISTORY: 2017-01-26 - Created
#

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import pdb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(127)

#Setup image parameters
img_width = 24
img_height = 24
batch_size = 32

#Setup directories
train_data_dir = '/home/kikimei/Insight/esp/data/train'
validation_data_dir = '/home/kikimei/Insight/esp/data/validation'
nb_train_samples = 3940 #1970 total training samples
nb_val_samples = 906 #400 total samples
nb_epoch = 9

# Get Training Data
xtrain = np.load('../openclosed_train_nn.dat.npy')
ytrain = np.load('../openclosed_class_train_nn.dat.npy')

# Get Validation Data
xvalid = np.load('../openclosed_val_nn.dat.npy')
yvalid = np.load('../openclosed_class_val_nn.dat.npy')

#rotation_range=40,
#width_shift_range=0.2,
#height_shift_range=0.2,

# add inception layer == border_mode='same' - prevent from losing edges
# + more convolutions to get down to 1

model = Sequential()
#remove border_mode=same
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
# 24 -> 12
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

# 12 -> 6
model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

#6 --> 3
model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

# 3 -> 1
#model.add(Convolution2D(32,3,3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#model.add(Activation('softmax'))

#pdb.set_trace()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Augmenting the training dataset by doing image transformations on the
#limited dataset
# zoom+shear
#train_datagen = ImageDataGenerator(
#    rescale=1./255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    fill_mode='nearest')
#
##Augmenting the testing dataset just has scaling
#test_datagen = ImageDataGenerator(rescale=1./255)
#
##Taking pictures from the training directories and generating
##augmented datasets for training
#train_gen = train_datagen.flow_from_directory(
#    train_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=32,
#    class_mode='binary')
#
## Same as above but for valiation directories
#validation_gen = test_datagen.flow_from_directory(
#    validation_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=32,
#    class_mode='binary')
#
#model.fit_generator(
#    train_gen,
#    samples_per_epoch=nb_train_samples,
#    nb_epoch=nb_epoch,
#    validation_data=validation_gen,
#    nb_val_samples=nb_val_samples)

model.fit(xtrain, ytrain,
          validation_split=0.2,
          batch_size=batch_size,
          nb_epoch=nb_epoch)

#Output model
model.save_weights('nn_model_20epochs.h5')

model_json = model.to_json()

pdb.set_trace()
with open("nn_model_20epochs.json", "w") as json_file:
    json_file.write(model_json)

score = model.evaluate(xvalid, yvalid)

print('accuracy: %.2f%%' % score[1]*100)

test = np.zeros((1,24,24,3))
test[0] = xvalid[0]

model.predict(test)

pdb.set_trace()

