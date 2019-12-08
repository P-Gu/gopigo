import sys
import keras
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.models import load_model
import numpy as np
import os, sys

import matplotlib.image as mpimg

from PIL import Image

model = load_model("my_model.h5")
X=[]
Y=[]

for root, dirs, files in os.walk("Data", topdown=True):
   for file in files:
      if os.path.basename(root)[-1] == "w":
          label = os.path.basename(root)[-1]
          img = Image.open(os.path.join(root,file))
          img = img.resize((227,227))
          img = np.asarray(img)
          img = img.astype(np.float32)
          img = (img / 127.5) - 1
          X.append(img)
          Y.append(label)
   if os.path.basename(root)[-1] == "w": break

X = np.asarray(X)
#im = np.expand_dims(img, axis=0)
y_prob = model.predict(X)
print(y_prob)
print(Y)