# data collection -- last edit Amanda 
from gopigo import *
import gopigo
import gopigo3
import sys
import pygame
import picamera
import os, os.path
import time
import csv
from easygopigo3 import EasyGoPiGo3

#inicialized camera:
camera = picamera.PiCamera()
clock = pygame.time.Clock()
# inicailied pygame:
pygame.init()
# init screen with background bg and text display
screen = pygame.display.set_mode((700,400))
pygame.display.set_caption('Car Remote Control Window')
bg = pygame.Surface(screen.get_size())
bg = bg.convert()
bg.fill((250,250,250))
text = "Basic GoPiGo control GUI"
font = pygame.font.Font(None,36)
displayText = font.render(text,1,(10,10,10))
bg.blit(displayText,(10,10))
screen.blit(bg,(0,0))
pygame.display.flip()

# w: fordward, a:left, d:right
# declear len of the name for each picture 
picID = 0
imgSize_w = len([name for name in os.listdir('/home/pi/Desktop/Project/Data/w') if os.path.isfile(os.path.join('/home/pi/Desktop/Project/Data/w',name))])
imgSize_a = len([name for name in os.listdir('/home/pi/Desktop/Project/Data/a') if os.path.isfile(os.path.join('/home/pi/Desktop/Project/Data/a',name))])
imgSize_d = len([name for name in os.listdir('/home/pi/Desktop/Project/Data/d') if os.path.isfile(os.path.join('/home/pi/Desktop/Project/Data/d',name))])
# data collection 
file = open("keyboard","w")
previous = ''
char = ''
acceleration = 0
# init gopigo driver:
GPG = gopigo3.GoPiGo3()
# record the result:
direction = list()
run = True

while run :
    # set up event 
    event = pygame.event.wait()
    pygame.event.clear()
    print("waiting for the event...")
    # check input and conidtion on each input:
    # gopigo.motorx(dir,speed): x -> 1 or 2 ; dir: 1 forward 0 back; speed: 0 stop 255 full speed
#    if pygame.key.get_focused() :
#        print("key pressed",pygame.event.peek())
#    print("Press the key")
#    clock.tick(1000)
#    print("Key received!")
    if event.type == pygame.KEYUP :
        gopigo.motor1(1,0) # forward with speed 0
        gopigo.motor2(1,0)
        char = 'n'
        continue
    elif event.type != pygame.KEYDOWN :
        char = 'n'
        continue
    else :
        print("event activated:",char)
        char = event.unicode
        direction.append(char)
        if char == 'w' :
            print("Taking picture and move forward in : w")
            imageName = '/home/pi/Desktop/Project/Data/result/' + str(picID)+'.jpg'
            picID = picID + 1
            camera.capture(imageName)
            for i in range(0, 80):
                GPG.set_motor_power(GPG.MOTOR_LEFT + GPG.MOTOR_RIGHT, i)
                time.sleep(0.02)
            GPG.reset_all()
            
        if char == 'a' :
            print("Taking picture and move forward in : a")
            imageName = '/home/pi/Desktop/Project/Data/result/' + str(picID)+'.jpg'
            picID = picID + 1
            camera.capture(imageName)
            for i in range(0,40):
                GPG.set_motor_power(GPG.MOTOR_RIGHT, i)
                time.sleep(0.02)
            GPG.reset_all()
            
        if char == 'd' :
            print("Taking picture and move forward in : d")
            imageName = '/home/pi/Desktop/Project/Data/result/' + str(picID)+'.jpg'
            picID = picID + 1
            camera.capture(imageName)
            for i in range(0,40):
                GPG.set_motor_power(GPG.MOTOR_LEFT, i)
            for i in range(0, 70):
                #GPG.set_motor_power(GPG.MOTOR_LEFT + GPG.MOTOR_RIGHT, i)
                time.sleep(0.02)
            GPG.reset_all()
            
        if char == 's' :
            print("Move backward : s")
            for i in range(-80, 0):
                GPG.set_motor_power(GPG.MOTOR_LEFT + GPG.MOTOR_RIGHT, i)
                time.sleep(0.02)
            GPG.reset_all()
            
        if char == 'q' :
            print("quiting the program...")
            print("direction result:")
            print(direction)
            result = zip(direction)
            output_name = "keyboard_result.csv"
            with open(output_name, "w", newline = "") as csvfile :
                writer = csv.writer(csvfile)
                writer.writerows(result)
            sys.exit()

#    for event in pygame.event.get():
#        if event.type == pygame.KEYDOWN :
#            if event.key == pygame.K_w :
#                print("go to w")
        
        

#def main():
#    #inicialized program
#    init()
#    # take picture:
#    takePicture()
#    
#
#
#if __name__ == '__main__' :
#    main()
