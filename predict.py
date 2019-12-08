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
import csv

import matplotlib.image as mpimg

from PIL import Image

model = load_model("my_model2.h5")
X=[]
Y=[]

for root, dirs, files in os.walk("Dec7DataModified", topdown=True):
   for file in files:
      if os.path.basename(root)[-1] == "a" and len(os.path.basename(root))==1:
          label = os.path.basename(root)[-1]
          img = Image.open(os.path.join(root,file))
          img = img.resize((227,227))
          img = np.asarray(img)
          img = img.astype(np.float32)
          img = (img / 127.5) - 1
          X.append(img)
          Y.append(label)
   if os.path.basename(root)[-1] == "a" and len(os.path.basename(root))==1: break

X = np.asarray(X)
#im = np.expand_dims(img, axis=0)
y_prob = model.predict(X)
print(y_prob[:,0])

labelList = ["d", "w", "a"]

Y_prob = []

for i in range(y_prob.shape[0]):
    row = y_prob[i, : ]
    index = np.argmax(row)
    label = labelList[index]
    Y_prob.append(label)

print(Y)

with open('result.csv', 'w') as myfile:
    wr = csv.writer(myfile, dialect='excel')
    wr.writerows(Y_prob)