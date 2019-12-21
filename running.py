from gopigo import *
import gopigo
import gopigo3
import sys
import pygame
import picamera
import os, os.path
import time
import csv
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
from keras.models import load_model
import h5py
from easygopigo3 import EasyGoPiGo3
from PIL import Image
import numpy as np

### Trying to make sure the keras don't fail
"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)"""

# take picture
picture_id = 0
char = ""
try :
#while True: 
    camera = picamera.PiCamera()
    imageName = '/home/pi/Desktop/Project/Data/result/' + str(picture_id)+'.jpg'
    camera.capture(imageName)
    print("finished")
    Image.open(imageName)
    picture_id = picture_id + 1
    # get the train model and make prediction
    #model = load_model("my_model.h5")
    model = h5py.File("my_model.h5", "r")
    X=[]
    img = Image.open(imageName)
    img = img.resize((227,227))
    img = np.asarray(img)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1
    X.append(img)

    X = np.asarray(X)
    #im = np.expand_dims(img, axis=0)
    #y_prob = model.predict(X)
    #print(y_prob)
    # move
    if char == 'w' :
        print("Move: w")
        for i in range(0, 80):
            GPG.set_motor_power(GPG.MOTOR_LEFT + GPG.MOTOR_RIGHT, i)
            time.sleep(0.02)
        GPG.reset_all()
        
    if char == 'a' :
        print("Move : a")
        for i in range(0,20):
            GPG.set_motor_power(GPG.MOTOR_RIGHT, i)
            time.sleep(0.02)
        GPG.reset_all()
        
    if char == 'd' :
        print("Move : d")
        for i in range(0,20):
            GPG.set_motor_power(GPG.MOTOR_LEFT, i)
            time.sleep(0.02)
        GPG.reset_all()
        
    print("Finished")

except KeyboardInterrupt:
    GPG.reset_all()
    os.exit(0)
# accuracy
