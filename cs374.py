import sys
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import numpy as np
import os, sys

from pathlib import Path
from matplotlib.image import imread
import cv2
from scipy import misc
#from skimage.transform import resize
#from skimage import io

from sklearn.utils import shuffle

np.random.seed(1000)
trainPercent = 0.75
X = []
Y = []
resize = 227
labelSet = set()
count=0

for root, dirs, files in os.walk("photos", topdown=True):
   print(root)
   for file in files:
       if file[-3:] == "png":
          label = os.path.basename(root)[4:]
          labelSet.add(label)
          img = cv2.imread(os.path.join(root,file))
          h = img.shape[0]
          w = img.shape[1]
          img = cv2.resize(img, (resize, resize))
          img = img.astype(np.float32)
          img = (img / 127.5) - 1
          X.append(img)
          Y.append(label)

trainPercent = 0.75

X_train = X[:int(X.shape[0]*trainPercent),:,:,:]
X_test = X[int(X.shape[0]*trainPercent):,:,:,:]
Y_train = Y[:int(Y.shape[0]*trainPercent),:]
Y_test = Y[int(Y.shape[0]*trainPercent):,:]

#%%capture output

#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11), strides=(4,4), padding='valid', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Dropout(0.5))
# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(227*227*3,), kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))

# 2nd Fully Connected Layer
model.add(Dense(4096, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))

# 3rd Fully Connected Layer
model.add(Dense(1000, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))

# Output Layer
model.add(Dense(mmpY.shape[1], kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('softmax'))

model.summary()

# Compile the model
opt = Adadelta(learning_rate = 0.005, rho = 0.95)
model.compile(loss=keras.losses.mean_squared_error, optimizer=opt, metrics=["accuracy"])

# 9. Fit model on training data
history = model.fit(X_train, Y_train, batch_size = 150, epochs=1000, verbose=2)


# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

print("score: "+str(score))