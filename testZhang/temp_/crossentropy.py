import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
import keras.backend as K
import numpy as np
cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
y_true = np.random.rand(50,4,4,313)
y_pred = np.random.rand(50,4,4,313)

# loss=cce(y_true,y_pred)

# print(loss)



# prior_factor = np.load("data/prior_factor.npy")
prior_factor = np.random.rand(313,1)
# prior_factor = prior_factor.astype(np.float32)

def categorical_mine(y_true, y_pred):
    q = 313
    y_true = K.reshape(y_true, (-1, q)) #Nxhxw, 313
    y_pred = K.reshape(y_pred, (-1, q)) #Nxhxw, 313

    cross_ent = K.categorical_crossentropy(y_pred, y_true) #Nxhxw,1

    idx_max = K.argmax(y_true, axis=1)
    weights = K.gather(prior_factor, idx_max)
    weights = K.reshape(weights, (-1, 1)) # Nhw x 1

    cross_ent = cross_ent * weights
    cross_ent = K.mean(cross_ent, axis=-1)
    print("cross entropy loss  shape",cross_ent.shape)
    return cross_ent

loss=categorical_mine(y_true,y_pred)

print(loss)