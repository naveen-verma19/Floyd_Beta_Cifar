import argparse

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import patience, epochs, num_train_samples, num_valid_samples, batch_size
from data_generator import train_gen, valid_gen
from model import build_model,build_simple_model
# from utils import get_available_gpus, categorical_crossentropy_color
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
import keras.backend as K
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')

if __name__ == '__main__':

    fpath1 = "/Users/naveen/Documents/tensor_test2/testZhang/model_saved.h5"
    fpath2="/Users/naveen/Documents/tensor_test2/testZhang/model_saved_val.h5"
    model_checkpoint = ModelCheckpoint(fpath1, monitor='loss', verbose=1, save_best_only=True)
    model_checkpoint2 = ModelCheckpoint(fpath2, monitor='val_loss', verbose=1, save_best_only=True)

    prior_factor = np.load("/Users/naveen/Documents/tensor_test2/testZhang/prior_factor.npy")
    prior_factor = K.cast(prior_factor, dtype='float32')

    def categorical_mine(y_true, y_pred):
        #ytrue= Bxhxwx313
        #ypred= Bxhxwx313 from model output

        # must return Bxhxw losses
        q = 313
        # y_true = K.reshape(y_true, (-1, q))  # Nxhxw, 313
        # y_pred = K.reshape(y_pred, (-1, q))  # Nxhxw, 313

        # cross_ent = tf.keras.losses.categorical_crossentropy(y_true,y_pred)  # Bxhxw losses per pixel
        cross_ent = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true,y_pred)
        idx_max = K.argmax(y_true, axis=-1)  #Bxhxw

        weights = K.gather(prior_factor, idx_max)
        ## wights same dim as idx_max

        # multiply cross_ent loss per pixel by weights
        cross_ent = cross_ent * weights
        # cross_ent = K.mean(cross_ent, axis=-1)  # must return Nxhxw size losses
        # multiply cross_ent loss per pixel by weights
        print("ytrue shape",y_true.shape,"y_pred shape",y_pred.shape,"cross ent shape",cross_ent.shape,"idx shape", idx_max.shape,
              "weights",weights.shape)

        '''Incompatible shapes: [16,32,313]weights vs. [16,32,32]crossent
        '''
        # K.reshape(cross_ent,(16,32,32))
        return cross_ent


    def categorical_mine_flatten(y_true, y_pred):
        # ytrue= Bxhxwx313
        # ypred= Bxhxwx313 from model output

        # must return Bxhxw losses
        q = 313
        y_true = K.reshape(y_true, (-1, q))  # Bxhxw, 313
        y_pred = K.reshape(y_pred, (-1, q))  # Bxhxw, 313

        cross_ent = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred) # Bxhxw

        idx_max = K.argmax(y_true, axis=1) # Bhw

        weights = K.gather(prior_factor, idx_max)
        # weights = K.reshape(weights, (-1, 1))  # Bhw x 1

        # multiply cross_ent loss per pixel by weights
        # cross_ent = cross_ent * weights
        # cross_ent = K.mean(cross_ent, axis=-1)  # must return Nxhxw size losses
        # multiply cross_ent loss per pixel by weights
        print("ytrue shape", y_true.shape, "y_pred shape", y_pred.shape, "cross ent shape", cross_ent.shape,
              "idx shape", idx_max.shape,
              "weights", weights.shape)

        '''Incompatible
        shapes: [160256, 1]  == 16*32*313,1
        vs.[16, 32, 32]
        '''
        # K.reshape(cross_ent,(16,32,32))
        return cross_ent

    new_model = build_simple_model()
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=5.)
    new_model.compile(optimizer='adam', loss=categorical_mine)

    new_model.load_weights(fpath1)
    # new_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print(new_model.summary())

    # Final callbacks
    callbacks = [model_checkpoint,model_checkpoint2]

    # Start Fine-tuning
    new_model.fit(train_gen(),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(),
                            validation_steps=num_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            )
    # new_model.fit_generator(train_gen(),
    #                         steps_per_epoch=num_train_samples // batch_size,
    #                         validation_data=valid_gen(),
    #                         validation_steps=num_valid_samples // batch_size,
    #                         epochs=epochs,
    #                         verbose=1,
    #                         callbacks=callbacks,
    #                         use_multiprocessing=True,
    #                         workers=8
    #                         )
