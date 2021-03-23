# %%
import keras.callbacks
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os

from tensorflow.python.framework.ops import disable_eager_execution
from matplotlib import pyplot as plt
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute

mlcompute.set_mlc_device(device_name='gpu')
from IPython.display import clear_output
# %%
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()


plot_losses = PlotLosses()

factor = int(256 / 128)

# Get images
train_folder_cifar = "/Users/naveen/Downloads/Celeb 128x128/"
def get_XY():
    # train_folder_cifar = '/Users/naveen/Documents/tensor_test2/cifar_train/'
    files_sorted = os.listdir(train_folder_cifar)
    files_sorted = [x for x in files_sorted if x.endswith(".jpg")]
    files_sorted.sort(key=lambda x: int(x.split(".")[0]))
    X = []
    for filename in files_sorted[:6000]:  # you can train on the whole 50000 if you want
        X.append(img_to_array(load_img(train_folder_cifar + filename)))
    X = np.array(X, dtype=float)

    # Set up train and test data
    split = int(0.95 * len(X))
    Xtrain = X[:split]

    # splitting validation
    Xtrain = 1.0 / 255 * Xtrain
    split_idx = int(0.90 * len(Xtrain))
    Xvalid = Xtrain[split_idx:]
    Xtrain = Xtrain[:split_idx]

    return Xtrain,Xvalid,files_sorted

def get_XY_places():
    path_prefix ="/Users/naveen/Documents/ML local Data/"
    fnames=open("/Users/naveen/Documents/tensor_test2/places_top10_local","r").read().split("\n")
    fnames=[path_prefix+x.split(" ")[0] for x in fnames]
    X = []
    Xv=[]
    test_files=[]
    i=0
    st=4000
    for folder in fnames:
        i+=1
        files_sorted = os.listdir(folder)
        files_sorted = [x for x in files_sorted if x.endswith(".jpg")]
        files_sorted.sort(key=lambda x: int(x.split(".")[0]))
        j=0
        for filename in files_sorted[:st]:  # you can train on the whole 50000 if you want
            print(i,j)
            j+=1
            X.append(img_to_array(load_img(folder + filename)))
        for filename in files_sorted[st:st+200]:
            Xv.append(img_to_array(load_img(folder + filename)))
            print(i,j)
            j += 1
        for filename in files_sorted[st+200:st+350]:
            test_files.append(folder+filename)
            print(i,j)
            j += 1

    X = np.array(X, dtype=float)
    Xv = np.array(Xv, dtype=float)
    np.random.shuffle(X)
    np.random.shuffle(Xv)
    # Set up train and test data
    # split = int(0.95 * len(X))
    #
    # # splitting validation
    X = 1.0 / 255 * X
    Xv = 1.0 / 255*Xv
    # split_idx = int(0.90 * len(Xtrain))
    # Xvalid = Xtrain[split_idx:]
    # Xtrain = Xtrain[:split_idx]

    return X,Xv,test_files

Xtrain,Xvalid,test_files= get_XY_places()

# Xtrain,Xvalid,files_sorted=get_XY()
print("images loaded into ram")
# %%
model = Sequential()
model.add(InputLayer(input_shape=(int(256/2), int(256/2), 1)))
model.add(Conv2D(int(64 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
model.add(Conv2D(int(64 / factor), (3, 3), activation='relu', padding='same', strides=2, use_bias=True))

model.add(Conv2D(int(128 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
model.add(Conv2D(int(128 / factor), (3, 3), activation='relu', padding='same', strides=2, use_bias=True))
# model.add(BatchNormalization())

model.add(Conv2D(int(256 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
model.add(Conv2D(int(256 / factor), (3, 3), activation='relu', padding='same', strides=2))
# model.add(BatchNormalization())

model.add(Conv2D(int(512 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
model.add(Conv2D(int(256 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
model.add(Conv2D(int(128 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
# model.add(BatchNormalization())

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(int(64 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(int(32 / factor), (3, 3), activation='relu', padding='same', use_bias=True))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same', use_bias=True))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')

# Image transformer
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True)

# Generate training data
batch_size = 100


def train_generator(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:, :, :, 0]
        Y_batch = lab_batch[:, :, :, 1:] / 128
        yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)


def validationGenerator(batch_size):
    for batch in datagen.flow(Xvalid, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:, :, :, 0]
        Y_batch = lab_batch[:, :, :, 1:] / 128
        yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)


# Train model
filepath = "/Users/naveen/Documents/tensor_test2/model.h5"
filepath2 = "/Users/naveen/Documents/tensor_test2/model_val.h5"

# load the the best so far model if already trained
# model.load_weights(filepath)

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
callbacks_list = [es, checkpoint,checkpoint2]

steps_per_epoch = int(len(Xtrain) / batch_size)  # so that 1 epoch passes through whole data
validation_steps = int(len(Xvalid) / batch_size)
# model.fit(train_generator(batch_size), callbacks=callbacks_list, epochs=1000, steps_per_epoch=steps_per_epoch,validation_data=validationGenerator(batch_size), validation_steps=validation_steps)
# load best model for testing
model.load_weights(filepath)

def test_places():
    color_me = []
    for f in test_files:
        color_me.append(img_to_array(load_img(f)))
    color_me = np.array(color_me, dtype=float)
    color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape + (1,))

    # Test model
    print("testing...")
    output = model.predict(color_me)
    output = output * 128
    # Output colorizations
    for i in range(len(output)):
        print(i)
        cur = np.zeros((int(256 / factor), int(256 / factor), 3))
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        imsave("places_out/" + str(4200+i+1) + "a.png", lab2rgb(cur))

test_places()

# test the training data and saves the images
# def test_train():
#     color_me = []
#     for filename in files_sorted[:500]:
#         fname = train_folder_cifar + filename
#         color_me.append(img_to_array(load_img(fname)))
#     color_me = np.array(color_me, dtype=float)
#     color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
#     color_me = color_me.reshape(color_me.shape + (1,))
#
#     # Test model
#     print("testing...")
#     output = model.predict(color_me)
#     output = output * 128
#
#     # Output colorizations
#     for i in range(len(output)):
#         print(i)
#         input_file = files_sorted[i].split(".")[0]
#         cur = np.zeros((int(256 / factor), int(256 / factor), 3))
#         cur[:, :, 0] = color_me[i][:, :, 0]
#         cur[:, :, 1:] = output[i]
#         imsave("results/" + input_file + ".png", lab2rgb(cur))
#
# def testResults():
#     color_me = []
#     # should be after your training data
#     st=12000
#     for i in range(st, st+500):
#         filename = files_sorted[i]
#         fname = train_folder_cifar + filename
#         color_me.append(img_to_array(load_img(fname)))
#     color_me = np.array(color_me, dtype=float)
#     color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
#     color_me = color_me.reshape(color_me.shape + (1,))
#
#     # Test model
#     output = model.predict(color_me)
#     output = output * 128
#
#     # Output colorizations
#     for i in range(len(output)):
#         cur = np.zeros((int(256 / factor), int(256 / factor), 3))
#         cur[:, :, 0] = color_me[i][:, :, 0]
#         cur[:, :, 1:] = output[i]
#         imsave("results2/" + str(i + st + 1) + ".png", lab2rgb(cur))

#
# def testGraySingle():
#     color_me = []
#     #
#     # filename=files_sorted[i]
#     # fname = train_folder_cifar + filename
#     color_me.append(img_to_array(load_img("testgray2.jpg")))
#
#     color_me = np.array(color_me, dtype=float)
#     color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
#     color_me = color_me.reshape(color_me.shape + (1,))
#
#     # Test model
#     output = model.predict(color_me)
#     output = output * 128
#
#     # Output colorizations
#     for i in range(len(output)):
#         cur = np.zeros((int(256 / factor), int(256 / factor), 3))
#         cur[:, :, 0] = color_me[i][:, :, 0]
#         cur[:, :, 1:] = output[i]
#         imsave("results3/" + str(i + 3) + ".png", lab2rgb(cur))


# test_train()
# testResults()
# testGraySingle()


