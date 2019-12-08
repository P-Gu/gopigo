import sys
import keras
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import numpy as np
import os, sys

import matplotlib.image as mpimg

from PIL import Image
import cv2
from sklearn.utils import shuffle

np.random.seed(1000)
trainPercent = 0.75
X = []
Y = []
resize = 227
labelSet = set()
count=0
wCount = 0

for root, dirs, files in os.walk("Data", topdown=True):
   for file in files:
      if os.path.basename(root)[-2:] != "ta":
          if os.path.basename(root)[-1] == "w": wCount += 1
          if wCount >= 40 and os.path.basename(root)[-1] == "w": continue
          label = os.path.basename(root)[-1]
          labelSet.add(label)
          #img = cv2.imread(os.path.join(root,file))
          img = Image.open(os.path.join(root,file))
          #if label == "a":
          #    mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
          #    mirror = mirror.resize((resize, resize))
          #    mirror = np.asarray(mirror)
          #    mirror = mirror.astype(np.float32)
          #    mirror = (mirror / 127.5) - 1
          #    X.append(mirror)
          #    Y.append("d")
          #if label == "d":
          #    mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
          #    mirror = mirror.resize((resize, resize))
          #    mirror = np.asarray(mirror)
          #    mirror = mirror.astype(np.float32)
          #    mirror = (mirror / 127.5) - 1
          #    X.append(mirror)
          #    Y.append("a")
          #img = cv2.resize(img, (resize, resize))
          img = img.resize((resize,resize))
          img = np.asarray(img)
          img = img.astype(np.float32)
          img = (img / 127.5) - 1
          X.append(img)
          Y.append(label)


labelList = list(labelSet)
for i in range(len(Y)):
   #This is not one-hot
   l = [0]*len(labelList)
   index = labelList.index(Y[i])
   l[index] = 1
   Y[i] = l


X = np.asarray(X)
Y = np.asarray(Y)

p = np.random.permutation(len(X))
X = X[p]
Y = Y[p]

print(Y)

trainPercent = 0.75

X_train = X[:int(X.shape[0]*trainPercent),:,:,:]
X_test = X[int(X.shape[0]*trainPercent):,:,:,:]
Y_train = Y[:int(Y.shape[0]*trainPercent),:]
Y_test = Y[int(Y.shape[0]*trainPercent):,:]

#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11), strides=(4,4), padding='valid'))#, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))#, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))#, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))#, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))#, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Dropout(0.5))
# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(227*227*3,)))#, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))

# 2nd Fully Connected Layer
model.add(Dense(4096))#, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))

# 3rd Fully Connected Layer
model.add(Dense(1000))#, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))

# Output Layer
model.add(Dense(3))#, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('softmax'))

model.summary()

# Compile the model
opt = Adadelta(learning_rate = 0.005)#, rho = 0.95)
model.compile(loss=keras.losses.mean_squared_error, optimizer=opt, metrics=["accuracy"])

# 9. Fit model on training data
history = model.fit(X_train, Y_train, batch_size = 64, epochs=50, verbose=2)


# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

print("score: "+str(score))

model.save('my_model2.h5')