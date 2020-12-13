## Import all required variables
import os
import cv2
import time
import busio

import numpy as np
import tensorflow as tf
from tensorflow import keras

from board import SDA, SCL
from adafruit_pca9685 import PCA9685


##################Hand Class############################################
class partHand():
	## Declare Object Variables
	def __init__(self, obj):
		self.index = 0
		self.middle = 0
		self.ring = 0
		self.little = 0
		self.thumb = 0
		self.fist = 120
		self.frequency = 75
		self.mod = obj
		self.f1 = obj.channels[0]
		self.f2 = obj.channels[1]
		self.f3 = obj.channels[2]
		self.f4 = obj.channels[3]
		self.f5 = obj.channels[4]
		self.f6 = obj.channels[5]
		self.start = 2000
		self.end = 9000
		self.maxAngle = 180

	## Write finger values to corresponding motor
	def writeMod(self):
		self.angle = self.index
		self.f1.duty_cycle = int(self.calcValue())
		self.angle= self.middle
		self.f2.duty_cycle = int(self.calcValue())
		self.angle= self.ring
		self.f3.duty_cycle = int(self.calcValue())
		self.angle= self.little
		self.f4.duty_cycle = int(self.calcValue())
		self.angle= self.thumb
		self.f5.duty_cycle = int(self.calcValue())
		self.angle= self.fist
		self.f6.duty_cycle = int(self.calcValue())
		pass

	## Calculate values to pass to motor according to required angle
	def calcValue(self):
		if self.angle < 0:
			self.angle = 0
		elif self.angle > 180:
			self.angle = 180
		return (((self.end - self.start) * self.angle) / self.maxAngle) + self.start
		pass

	## Set PWM frequency
	def setFreq(self):
		self.mod.frequency = self.frequency
		pass

#############Gesture Functions######################
	def zero(self):
		self.index = 180
		self.middle = 180
		self.ring = 180
		self.little = 180
		self.thumb = 180
		pass

	def one(self):
		self.index = 0
		self.middle = 180
		self.ring = 180
		self.little = 180
		self.thumb = 180
		pass

	def two(self):
		self.index = 0
		self.middle = 0
		self.ring = 180
		self.little = 180
		self.thumb = 180
		pass

	def three(self):
		self.index = 0
		self.middle = 0
		self.ring = 0
		self.little = 180
		self.thumb = 180
		pass

	def four(self):
		self.index = 0
		self.middle = 0
		self.ring = 0
		self.little = 0
		self.thumb = 180
		pass

	def five(self):
		self.index = 0
		self.middle = 0
		self.ring = 0
		self.little = 0
		self.thumb = 0
		pass

	def thumbsup(self):
		self.index = 180
		self.middle = 180
		self.ring = 180
		self.little = 180
		self.thumb = 0
		pass

	def spiderman(self):
		self.index = 0
		self.middle = 180
		self.ring = 180
		self.little = 0
		self.thumb = 0
		pass

	def rock(self):
		self.index = 0
		self.middle = 180
		self.ring = 180
		self.little = 0
		self.thumb = 180
		pass

	def ok(self):
		self.index = 180
		self.middle = 0
		self.ring = 0
		self.little = 0
		self.thumb = 180
		pass
####################################################

########################################################################

############# Helper Functions ################
## Get region of intrest from frame
def getFrames(frame):
    shape_fr = frame.shape
    start_pt = (int(shape_fr[1]/16),int(shape_fr[0]/4))
    end_pt = (int(shape_fr[1]/2-shape_fr[1]/16),int(shape_fr[0] - shape_fr[0]/4))

    new_frame = frame[int(shape_fr[0]/4):int(shape_fr[0] - shape_fr[0]/4),int(shape_fr[1]/16):int(shape_fr[1]/2-shape_fr[1]/16)]
    frame = cv2.rectangle(frame, start_pt, end_pt, (255,0,0), 2)

    return frame, new_frame
    pass

## Pre process the image for prediction
def preProcess(image):
	reszImg = cv2.resize(image, (224, 224))
	reszImg = cv2.cvtColor(reszImg, cv2.COLOR_BGR2RGB)
	preProsImg = tf.keras.applications.mobilenet.preprocess_input(reszImg)
	reshapImg = preProsImg.reshape(-1,224,224,3)
	return reshapImg, preProsImg
	pass

## Predict the image with trained neural network model
def getPredict(img, interpreter, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    outputData = interpreter.get_tensor(output_details[0]['index'])
    return outputData
    pass

## Change the brightness of image as required
def chBright(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2. split(hsv)
    lim = 255 - value
    v[v>lim] = 255
    v[v<=lim] += value
    final_hsv = cv2.merge((h,s,v))
    nwimg = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return nwimg
##############################################

####### Major Funcrions ###########

## Setup to initialize I2C 
## communication and class to control motors
def setup():
	bus = busio.I2C(SCL, SDA)
	pca = PCA9685(bus)
	clasObj = partHand(pca)
	clasObj.frequency = 60
	clasObj.setFreq()
	clasObj.writeMod()
	print("Setup Completed")
	time.sleep(3)
	return clasObj

## Main Function
def main():
	#in main
	hand = setup()

	predCount = 0
	preGest = "Zero"
	finalGesture = "Five"

	## Initialize the models 
	interpreter1 = tf.lite.Interpreter(model_path="mainModel.tflite")
	interpreter1.allocate_tensors()
	input_details1 = interpreter1.get_input_details()
	output_details1 = interpreter1.get_output_details()
	print('Model1 Loaded')

	## Setup the camera
	vid = cv2.VideoCapture(0)
	## Repeat until exit- 'q' is pressed
	while True:
		## Read Camera
		ret, frame = vid.read()
		if(ret == 0):
			break

		## Get ROI and process the frame and classify
		origFrame, roiFrame = getFrames(frame)
		#roiFrame = chBright(roiFrame, 50)
		predFrame, prosImg = preProcess(roiFrame)
		prediction = getPredict(predFrame, interpreter1, input_details1, output_details1)
		
		labels1 = ["Zero","One","Two","Three","Four","Five","Thumbs Up","Ok","Spider-man","Rock"]
		gest = labels1[np.argmax(prediction[0])]

		## Show Detected gesture in video window frame
		cv2.putText(frame, gest, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
		cv2.imshow('Video', frame)
		cv2.imshow('Prossed Image', prosImg)

		## Select final gesture after detecting 5 consecutive same gesture
		if preGest is not gest:
			predCount = 0
			preGest = gest
			pass
		else:
			predCount = predCount + 1
			if predCount >= 5:
				finalGesture = gest
				print(finalGesture)
				pass
			
			## Write the gestures to the motor
			if finalGesture == 'Zero':
				hand.zero()
			elif finalGesture == "One":
				hand.one()
			elif finalGesture == "Two":
				hand.two()
			elif finalGesture == "Three":
				hand.three()
			elif finalGesture == "Four":
				hand.four()
			elif finalGesture == "Five":
				hand.five()
			elif finalGesture == "Thumbs Up":
				hand.thumbsup()
			elif finalGesture == "Spider-man":
				hand.spiderman()
			elif finalGesture == "Rock":
				hand.rock()
			elif finalGesture == "Ok":
				hand.ok()
			hand.writeMod()
			pass

		## Check if keyboard key is hit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		pass
	
	## Release the video control and destory all windows
	vid.release()
	cv2.destroyAllWindows()

## If the program starts executing in main
if __name__ == "__main__":
	main()
