#%%

from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')

#%%

train_folder_cifar='cifar_train/'
# train_folder_cifar="/Users/naveen/Downloads/Celeb 128x128/"
factor=int(256/32)
files_sorted=os.listdir(train_folder_cifar)
files_sorted = [x for x in files_sorted if x.endswith(".png")]
files_sorted.sort(key=lambda x:int(x.split(".")[0]))


# Get images
X = []
for filename in files_sorted[:6000]: #you can train on the whole 50000 if you want
    X.append(img_to_array(load_img(train_folder_cifar+filename)))
X = np.array(X, dtype=float)

# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]

#splitting validation
Xtrain = 1.0/255*Xtrain
split_idx = int(0.9*len(Xtrain))
Xvalid=Xtrain[split_idx:]
Xtrain=Xtrain[:split_idx]


#%%

model = Sequential()
model.add(InputLayer(input_shape=(int(256/factor), int(256/factor), 1)))
model.add(Conv2D(int(64/factor), (3, 3), activation='relu', padding='same', use_bias=True))
model.add(Conv2D(int(64/factor), (3, 3), activation='relu', padding='same', strides=2,use_bias=True))

model.add(Conv2D(int(128/factor), (3, 3), activation='relu', padding='same',use_bias=True))
model.add(Conv2D(int(128/factor), (3, 3), activation='relu', padding='same', strides=2,use_bias=True))
# model.add(BatchNormalization())

model.add(Conv2D(int(256/factor), (3, 3), activation='relu', padding='same',use_bias=True))
model.add(Conv2D(int(256/factor), (3, 3), activation='relu', padding='same', strides=2))
# model.add(BatchNormalization())

model.add(Conv2D(int(512/factor), (3, 3), activation='relu', padding='same',use_bias=True))
model.add(Conv2D(int(256/factor), (3, 3), activation='relu', padding='same',use_bias=True))
model.add(Conv2D(int(128/factor), (3, 3), activation='relu', padding='same',use_bias=True))
# model.add(BatchNormalization())

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(int(64/factor), (3, 3), activation='relu', padding='same',use_bias=True))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(int(32/factor), (3, 3), activation='relu', padding='same',use_bias=True))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same',use_bias=True))
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
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

def validationGenerator(batch_size):
    for batch in datagen.flow(Xvalid, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)


# Train model
filepath = "/Users/naveen/Documents/tensor_test2/model.h5"

# load the the best so far model if already trained

model.load_weights(filepath)

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
callbacks_list = [es,checkpoint]

steps_per_epoch=int(len(Xtrain)/batch_size)  #so that 1 epoch passes through whole data
validation_steps=int(len(Xvalid)/batch_size)
# model.fit(train_generator(batch_size),callbacks=callbacks_list,epochs=1000,steps_per_epoch=steps_per_epoch,validation_data=validationGenerator(batch_size), validation_steps=validation_steps)
#load best model for testing
model.load_weights(filepath)

#%%

#test the training data and saves the images
def test_train():
    color_me = []
    for filename in files_sorted[:500]:
        fname=train_folder_cifar+filename
        color_me.append(img_to_array(load_img(fname)))
    color_me = np.array(color_me, dtype=float)
    color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))

    # Test model
    print("testing...")
    output = model.predict(color_me)
    output = output * 128

    # Output colorizations
    for i in range(len(output)):
        print(i)
        input_file=files_sorted[i].split(".")[0]
        cur = np.zeros((int(256/factor), int(256/factor), 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        imsave("results/"+input_file+".png", lab2rgb(cur))


def testResults():
    color_me = []
    #should be after your training data
    for i in range(12000,12500):
        filename=files_sorted[i]
        fname = train_folder_cifar + filename
        color_me.append(img_to_array(load_img(fname)))
    color_me = np.array(color_me, dtype=float)
    color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape + (1,))

    # Test model
    output = model.predict(color_me)
    output = output * 128

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((int(256 / factor), int(256 / factor), 3))
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        imsave("cifar_results/" + str(i+12000+1) + ".png", lab2rgb(cur))

def testGraySingle():
    color_me = []
    #
    # filename=files_sorted[i]
    # fname = train_folder_cifar + filename
    color_me.append(img_to_array(load_img("testgray2.jpg")))

    color_me = np.array(color_me, dtype=float)
    color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape + (1,))

    # Test model
    output = model.predict(color_me)
    output = output * 128

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((int(256 / factor), int(256 / factor), 3))
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        imsave("results3/" + str(i+3) + ".png", lab2rgb(cur))
# test_train()
testResults()
# testGraySingle()
