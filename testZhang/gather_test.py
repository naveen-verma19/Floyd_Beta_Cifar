import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import patience, epochs, num_train_samples, num_valid_samples, batch_size
from data_generator import train_gen, valid_gen
from model import build_model,build_simple_model
# from utils import get_available_gpus, categorical_crossentropy_color
import numpy as np
import keras.backend as K


prior_factor = np.load("prior_factor.npy")
prior_factor = K.cast(prior_factor, dtype='float32')

idx_max = np.random.randint(313, size=(16, 32, 32))

a=K.gather(prior_factor,idx_max)

print("")