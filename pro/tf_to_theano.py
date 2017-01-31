import keras
from keras import backend as K
from keras.utils.np_utils import convert_kernel
from keras.models import load_model

model = load_model('nn_model_final.h5')

for layer in model.layers:
    if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
        original_w = K.get_value(layer.W)
        converted_w = convert_kernel(original_w)
        K.set_value(layer.W, converted_w)

model.save('nn_model_final_theano.h5')
