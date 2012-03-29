'''
Created on Feb 20, 2012
@author: bseastwo

Last modified on March 5, 2012 
@author: jasperry, hnphan, danelson
'''

import time

import FirewireVideo
import imgutil
import pipeline
import source
import cv, cv2
import numpy
import os
import optparse
import shelve
import pylab
import matplotlib.pyplot as plt
from dc1394 import *


class DarkCurrentCalibration(pipeline.ProcessObject):
	'''
	Calculates the dark current of a camera and saves the result to a file
	'''
	def __init__(self, input = None, shelf = None):
		pipeline.ProcessObject.__init__(self, input)
		self.image = None
		self.count = 0
		self.mean = None
		self.shelf = shelf
		
	def generateData(self):
		input = self.getInput(0).getData()
		
		if self.count == 0:
			self.image = numpy.zeros((input.shape[0],input.shape[1],input.shape[2],100))
		elif self.count < 150:
			self.image[...,self.count%100] = input
		else:
			self.mean = numpy.mean(self.image,3)
			self.shelf.place("dark_current", self.mean)
		
		self.count += 1
		self.getOutput(0).setData(input)
		

class DarkCurrentAcquisition(pipeline.ProcessObject):
	'''
	Calculates the dark current of a camera and saves the result to a file
	'''
	def __init__(self, input = None, shelf = None):
		pipeline.ProcessObject.__init__(self, input)
		self.shelf = shelf
		
	def generateData(self):
		input = self.getInput(0).getData()
		input = input - self.shelf.get("dark_current")
		input[input<0]=0
		self.getOutput(0).setData(input)
		
			
class FlatFieldCalibration(pipeline.ProcessObject):
	'''
	Calculates the flat field of a camera and saves the result to a file
	'''
	def __init__(self, input = None, shelf = None):
		pipeline.ProcessObject.__init__(self, input)
		self.image = None
		self.count = 0
		self.meanImg = None
		self.mu = None
		self.shelf = shelf
		self.dark = self.shelf.get("dark_current")
		
	def generateData(self):
		input = self.getInput(0).getData()
		
		if self.count == 0:
			self.image = numpy.zeros((input.shape[0],input.shape[1],input.shape[2],100))
		elif self.count < 150:
			# be careful of negative values especially in xyz
			self.image[...,self.count%100] = input - self.dark
			self.image[self.image<0]=0
		else:
			self.meanImg = numpy.mean(self.image,3)				# image
			self.mu = numpy.mean(self.meanImg) 					# scalar
			self.shelf.place("color_balance_raw", self.meanImg)
			self.shelf.place("flat_field", self.mu/self.meanImg)
		
		self.count += 1
		self.getOutput(0).setData(input)
		
		
class FlatFieldAcquisition(pipeline.ProcessObject):
	'''
	Calculates the flat field of a camera and saves the result to a file
	'''
	def __init__(self, input = None, shelf = None):
		pipeline.ProcessObject.__init__(self, input)
		self.shelf = shelf
		
	def generateData(self):
		input = self.getInput(0).getData()
		input = input * self.shelf.get("flat_field")
		input[input>255] = 255
		self.getOutput(0).setData(input)		


class ColorBalanceCalibration(pipeline.ProcessObject):
	'''
	Calculates the flat field of a camera and saves the result to a file
	'''
	def __init__(self, input = None, shelf = None, hasData = True):
		pipeline.ProcessObject.__init__(self, input)
		self.image = None
		self.count = 0
		self.shelf = shelf
		self.dark = self.shelf.get("dark_current")
		self.hasData = hasData
		
	def generateData(self):
		input = self.getInput(0).getData()
		
		# if we did not do flat field
		if self.hasData == False:
			if self.count == 0:
				self.image = numpy.zeros((input.shape[0],input.shape[1],input.shape[2],100))
			elif self.count < 150:
				self.image[...,self.count%100] = input - self.dark
				self.image[self.image<0]=0
			else:
				### save stuff to shelf
				pass
		else:
			if self.count == 150:		# so we only run this once
				mode = 'rgb'
				if mode == 'rgb':
					meanR = numpy.mean(self.shelf.get("color_balance_raw")[...,0])
					meanG = numpy.mean(self.shelf.get("color_balance_raw")[...,1])
					meanB = numpy.mean(self.shelf.get("color_balance_raw")[...,2])
					self.shelf.place("color_balance_r", meanG/meanR)
					#self.shelf.place("color_balance_g", meanG/meanG)		### green channel is the reference
					self.shelf.place("color_balance_b", meanG/meanB)
				elif mode == 'xyz':
					pass
		
		self.count += 1
		self.getOutput(0).setData(input)
		

class ColorBalanceAcquisition(pipeline.ProcessObject):
	'''
	Calculates the flat field of a camera and saves the result to a file
	'''
	def __init__(self, input = None, shelf = None, hasData = True):
		pipeline.ProcessObject.__init__(self, input)
		self.shelf = shelf
		
	def generateData(self):
		input = self.getInput(0).getData()
		input = input.astype(numpy.float64)
		
		mode = 'rgb'
		if mode == 'rgb':
			input[...,0] = input[...,0] * self.shelf.get("color_balance_r")
			input[...,2] = input[...,2] * self.shelf.get("color_balance_b")
			input[input>255] = 255 
		elif mode == 'xyz':
			pass
		
		self.getOutput(0).setData(input)
		

class NoiseEstimation(pipeline.ProcessObject):
	''' 
	Calculates the apparent Noise of the camera and plot as a pixelwise line graph
	'''
	
	def __init__(self, input = None, shelf = None):
		pipeline.ProcessObject.__init__(self, input)
		self.image = None
		self.count = 0
		self.stdev = None
		self.mean = None
		self.shelf = shelf
		
	def generateData(self):
		input = self.getInput(0).getData()
		
		if self.count == 0:
			self.image = numpy.zeros((input.shape[0],input.shape[1],input.shape[2],100))
		elif self.count < 150:
			self.image[...,self.count%100] = input
		else:
			temp = self.image[::8,::8]
			self.stdev = numpy.std(temp,3)				# stdev per pixel
			self.mean = numpy.mean(temp,3)				# mean value per pixel

			#self.shelf.place("noise_estimation", self.stdev)
			
			pylab.ion()
			pylab.plot(self.mean.flatten(), self.stdev.flatten(), "b.")
			pylab.show()
			pylab.draw()
			
		self.count += 1
		self.getOutput(0).setData(input)
				

class AffineIntensity(pipeline.ProcessObject):
	'''
	Adjusts contrast by a given scalar and calculates an offset to maintain
	'''
	def __init__(self, input = None):
		pipeline.ProcessObject.__init__(self, input)
		self.scale = 1.0
		self.offset = 0.0
		
	def generateData(self):
		input = self.getInput(0).getData()
		
		if input.dtype == numpy.uint8:
			# lookup table
			lookUp = numpy.arange(256)
			lookUp = (lookUp * self.scale) + self.offset
			lookUp[lookUp>255] = 255
			lookUp[lookUp<0] = 0
			output = lookUp[input]
		else:
			output = (input * self.scale) + self.offset
			output[output>255] = 255
			output[output<0] = 0

		self.getOutput(0).setData(output)
		
	def setScale(self,scale):
		self.scale = scale
		self.modified()

	def setOffset(self,offset):
		self.offset = offset
		self.modified()
		
	def getScale(self):
		return self.scale
		
	def getOffset(self):
		return self.offset
		
	def autoContrast(self):
		input = self.getInput(0).getData()
		Imin = input.min()
		Imax = input.max()
		self.scale = 255.0/(Imax-Imin)
		self.offset = -self.scale * Imin
		self.modified()

	def setGain(self, value):
		input = self.getInput(0).getData()
		Imin = -self.offset/self.scale
		Imax = (255 - self.offset)/self.scale
		Imid = (Imax+Imin)/2
		self.scale = value
		self.offset = Imid - (self.scale*255/2)
		self.modified()


class Display(pipeline.ProcessObject):
	
	def __init__(self, input = None, name = "pipeline"):
		pipeline.ProcessObject.__init__(self, input)
		cv2.namedWindow(name, cv.CV_WINDOW_NORMAL)
		self.name = name
		
	def generateData(self):
		input = self.getInput(0).getData()
		# output here so channels don't get flipped
		self.getOutput(0).setData(input)

		# Convert back to OpenCV BGR from RGB
		if input.ndim == 3 and input.shape[2] == 3:
			input = input[..., ::-1]
		
		cv2.imshow(self.name, input.astype(numpy.uint8))			


class Shelf:
	
	def __init__(self, filename = "settings"):
		self.d = shelve.open(filename)
	
	def place(self, key, val):
		self.d[key] = val
		
	def get(self, key):
		return self.d[key]
		
	def defaultCamSettings(self):
		self.d["camera_id"] = 0
		self.d["mode"] = FirewireVideo.DC1394_VIDEO_MODE_640x480_RGB8
		self.d["frame_rate"] = FirewireVideo.DC1394_FRAMERATE_15


class Histogram(pipeline.ProcessObject):
   
	def __init__(self, input=None):
		pipeline.ProcessObject.__init__(self, input)
		self.count = 0
	   
	def generateData(self):
		input = self.getInput(0).getData()
		
		if self.count % 100 == 0:
			plt.ion()
			temp = input[::20,::20]
			plt.subplot('311')
			n, bin, patches = plt.hist(temp[...,0].flatten(),255,facecolor='red',linewidth=0)
			plt.subplot('312')
			n, bin, patches = plt.hist(temp[...,1].flatten(),255, facecolor='green',linewidth=0)
			plt.subplot('313')
			n, bin, patches = plt.hist(temp[...,2].flatten(),255, facecolor='blue',linewidth=0)   
			plt.draw()
	
		self.count+=1
		self.getOutput(0).setData(input)   
				

def calibrationDisplay(source, display, loop=False):
	key = None
	frame = 0
	t0 = time.time()
	span = 30
	while key != 27:
		source.updatePlayMode()
		display.update()
		key = cv2.waitKey(10)
		
		if key >= 0:
			char = chr(key)
			print "Key: ", key, char
		
		if frame == 150:
			key == 27
			break
		
		frame += 1
		
		if frame % span == 0:
			t1 = time.time()
			print "{0:8.5f} fps".format(span / (t1 - t0))
			t0 = t1
			
			if loop:
				source.increment()


def calibration(shelf):
	print "Starting image calibration..."
	"""
	pipeSource = source.CameraFW(0, 
		FirewireVideo.DC1394_VIDEO_MODE_640x480_RGB8,
		FirewireVideo.DC1394_FRAMERATE_15)
	pipeSource.getCamera().setColorAbsolute(whiteBlue=1023, whiteRed=276)
	"""
	pipeSource = source.CameraCV()
		
	raw_input("About to begin dark current calibration. Please block light passage to the cameras. Press return to continue.")
	dark = DarkCurrentCalibration(pipeSource.getOutput(), shelf)
	display = Display(dark.getOutput(), "pipeline")
	calibrationDisplay(pipeSource, display)
	print "Dark current calibration complete."
	
	raw_input("About to begin flat field calibration. Please allow light to pass to cameras and place a clear slide on the stage. Press return to continue.")
	flat = FlatFieldCalibration(pipeSource.getOutput(), shelf)
	display = Display(flat.getOutput(), "pipeline")
	calibrationDisplay(pipeSource, display)
	print "Flat field calibration complete."
	
	input = raw_input("Color calibration will default to rgb space. Would you like to run it in xyz space instead? y/n\n")
	if input == 'y':
		color = ColorBalanceCalibration(pipeSource.getOutput(), shelf)
		display = Display(color.getOutput(), "pipeline")
		calibrationDisplay(pipeSource, display)
		print "Color calibration in xyz space complete."
	else:
		color = ColorBalanceCalibration(pipeSource.getOutput(), shelf)
		display = Display(color.getOutput(), "pipeline")
		calibrationDisplay(pipeSource, display)
		print "Color calibration in rgb space complete."

	raw_input("About to begin noise calibration. Please place a high contrast specimen on the state. Press return to continue.")
	noise = NoiseEstimation(pipeSource.getOutput(), shelf)
	display = Display(noise.getOutput(), "pipeline")
	calibrationDisplay(pipeSource, display)
	print "Image calibration complete."
	
	
	
def acquisition(shelf):
	print "Starting image acquisition..."
	"""
	pipeSource = source.CameraFW(0, 
		FirewireVideo.DC1394_VIDEO_MODE_640x480_RGB8,
		FirewireVideo.DC1394_FRAMERATE_15)
	pipeSource.getCamera().setColorAbsolute(whiteBlue=1023, whiteRed=276)
	"""
	pipeSource = source.CameraCV()
	
	dark = DarkCurrentAcquisition(pipeSource.getOutput(), shelf)
	flat = FlatFieldAcquisition(dark.getOutput(), shelf)
	color = ColorBalanceAcquisition(flat.getOutput(), shelf)
	affine = AffineIntensity(color.getOutput())
	
	displaySource = Display(pipeSource.getOutput(), "pipeline source")
	displayDark = Display(dark.getOutput(), "pipeline dark current")
	displayFlat = Display(flat.getOutput(), "pipeline flat field")
	displayColor = Display(color.getOutput(), "pipeline color balance")
	displayAffine = Display(affine.getOutput(), "pipeline affine")
	
	hist = False
	multiple = False
	L = []
	i=0
	key = None
	frame = 0
	t0 = time.time()
	span = 30
	while key != 27:
	
		pipeSource.updatePlayMode()
		displaySource.update()
		displayAffine.update()
		displayDark.update()
		displayFlat.update()
		displayColor.update()
		if hist == True:
			histogram.update()

		key = cv2.waitKey(10)
		
		if key >= 0:
			char = chr(key)
			print "Key: ", key, char
			if char == "-":
				affine.setScale(affine.getScale() * 0.95)
			elif char == "=":
				affine.setScale(affine.getScale() * 1.05)
			elif char == "[":
				affine.setOffset(affine.getOffset() - 5.0)
			elif char == "]":
				affine.setOffset(affine.getOffset() + 5.0)
			elif char == ";":
				affine.setGain(affine.getScale() * 0.95)
			elif char == "'":
				affine.setGain(affine.getScale() * 1.05)
			elif char == "a":
				affine.autoContrast()	
			elif char == "r":
				affine.setScale(1.0)
				affine.setOffset(0.0)
			elif char == "h":
				histogram = Histogram(affine.getOutput())
				hist = True
			elif char == "o": # one-time AE
				pipeSource.getCamera().settleAutoExposure()
			elif char == "e": # AE
				pipeSource.getCamera().setAutoExposure()
			elif char == ".": # longer exposure
				# update current feature values
				pipeSource.getCamera().updateFeatures()
				pipeSource.getCamera().setExposureAbsolute(pipeSource.getCamera().shutter*1.095)
			elif char == ",": # shorter exposure
				pipeSource.getCamera().updateFeatures()
				pipeSource.getCamera().setExposureAbsolute(pipeSource.getCamera().shutter/1.095)
			elif char == "s":
				name = "img-%04d.png" % (i)
				cv2.imwrite(name, displayColor.getInput().getData())
				i+=1
			elif char == "m":
				multiple = not multiple
				if not multiple:
					print len(L)
					for element in L:
						name = "img-%04d.png" % (i)
						cv2.imwrite(name, L[i])
						i += 1
					L = []
			
		if multiple:
			L.append(displayColor.getInput().getData())
				
		frame += 1
		if frame % span == 0:
			t1 = time.time()
			print "{0:8.5f} fps".format(span / (t1 - t0))
			t0 = t1
		
	
def main():
	parser = optparse.OptionParser()
	parser.add_option("-p", "--path", help="the path of the image folder", default=None)
	parser.add_option("-c", "--color", help="color mode for the camera", default=DC1394_COLOR_CODING_YUV422)
	parser.add_option("-s", "--size", help="size of the image", default=DC1394_VIDEO_MODE_1024x768_RGB8)
	parser.add_option("-f", "--framerate", help="framerate of the camera", default=0)
	options, remain = parser.parse_args()
	
	# display images from file
	if options.path != None:
		images = os.listdir(options.path)
		for i in range(len(images)):
			images[i] = options.path + images[i]
		pipeSource = source.FileStackReader(images)
		pipeSource.setLoop()
		
		display = Display(pipeSource.getOutput())
		
		opencvDisplay(pipesource, display, True)
		exit(0)
	

	input = raw_input("Do you want to use default camera settings? y/n\n")
	settings = Shelf("settings")
	
	if input == 'y':
		settings.defaultCamSettings()
		
	input = raw_input("Would you like to run image calibration? y/n\n")
	if input == 'y':
		calibration(settings)
		
	input = raw_input("Would you like to run image acquisition? y/n\n")
	if input == 'y':
		acquisition(settings)
	
	
	
if __name__ == "__main__":
	main()
