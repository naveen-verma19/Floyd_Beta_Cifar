import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from keras.models import Model
from keras.regularizers import l2
from keras.utils import plot_model
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer

from config import img_rows, img_cols, num_classes, kernel
from keras.models import Sequential, load_model

l2_reg = l2(1e-3)


def build_simple_model():
    factor = int(256 / 128)
    model = Sequential()
    model.add(InputLayer(input_shape=(img_rows, img_cols, 1)))
    #now 256x256x1
    model.add(Conv2D(int(64 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
    model.add(Conv2D(int(64 / factor), (3, 3), activation='relu', padding='same', strides=2, use_bias=True))
    # now 128x128x64

    model.add(Conv2D(int(128 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
    model.add(Conv2D(int(128 / factor), (3, 3), activation='relu', padding='same', strides=2, use_bias=True))
    # model.add(BatchNormalization())
    #now 64x64x128
    model.add(Conv2D(int(256 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
    model.add(Conv2D(int(256 / factor), (3, 3), activation='relu', padding='same', strides=2))
    # model.add(BatchNormalization())
    # now 32x32x256
    model.add(Conv2D(int(512 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
    # now 32x32x512
    model.add(Conv2D(int(256 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
    #now 32x32x256
    # model.add(Conv2D(int(128 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
    # model.add(BatchNormalization())

    model.add(UpSampling2D((2, 2)))
    # now 64x64x256

    outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same', name='pred')
    model.add(outputs)
    #now 64x64x313  [0-1]

    # model.add(Conv2D(int(64 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(int(32 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
    # model.add(Conv2D(2, (3, 3), activation='tanh', padding='same', use_bias=True))  # tanh
    # model.add(UpSampling2D((2, 2)))
    return model
def build_model():
    input_tensor = Input(shape=(img_rows, img_cols, 1))
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(input_tensor)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg,
               strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_3', kernel_initializer="he_normal",
               strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_3',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_3',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_3',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_3',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_3',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same', name='pred')(x)

    model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
    return model

