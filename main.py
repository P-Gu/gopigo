from gopigo import *
import gopigo
import gopigo3
import sys
import pygame
import picamera
import os, os.path
import time
import csv

GPG = gopigo3.GoPiGo3()
with open('true1.csv') as csvfile :
	readCSV = csv.reader(csvfile, delimiter = '\n')
	for letter in readCSV :
		array = letter
		char = array[0]
		print(char)
		if char == 'w' :
			print("Move: w")
			for i in range(0, 80):
				GPG.set_motor_power(GPG.MOTOR_LEFT + GPG.MOTOR_RIGHT, i)
				time.sleep(0.02)
			GPG.reset_all()
        
		if char == 'a' :
			print("Move : a")
			for i in range(0,60):
				GPG.set_motor_power(GPG.MOTOR_RIGHT, i)
				time.sleep(0.02)
			GPG.reset_all()
        
		if char == 'd' :
			print("Move : d")
			for i in range(0,60):
				GPG.set_motor_power(GPG.MOTOR_LEFT, i)
				time.sleep(0.02)
			GPG.reset_all()
		
        else :
			print("Finished")
