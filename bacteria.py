'''
Created on Feb 20, 2012

@author: bseastwo
'''

import time

#import FirewireVideo
import imgutil
import pipeline
import source
import numpy
import os
import glob
import optparse
import cv2
import cv
from scipy import ndimage

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
            output = output.astype(numpy.uint8)

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


class BinarySegmentation(pipeline.ProcessObject):
    '''
    Segments the bacteria colonies in the images.
    '''
    def __init__(self, input = None, bgImg = None):
        pipeline.ProcessObject.__init__(self, input)
        self.bgImg = bgImg
        self.binary = numpy.zeros(bgImg.shape)
    
    def generateData(self):
        input = self.getInput(0).getData()
        #print "input: ",input[465,485]
        #print "bg: ", self.bgImg[465,485]
        output = (input.astype(numpy.float) - self.bgImg.astype(numpy.float)) #background subtraction
        output = output + 40
        print "output: ", output[694,713]

        tempBinary = numpy.zeros(output.shape)
        tempBinary[output < 30] = 1
        tempBinary[output >= 30] = 0

        tempBinary = ndimage.grey_erosion(tempBinary, size = (3,3))
        tempBinary = ndimage.grey_erosion(tempBinary, size = (3,3))
        tempBinary = ndimage.grey_dilation(tempBinary, size = (3,3))

        self.binary = numpy.logical_or(self.binary, tempBinary).astype(numpy.uint8)
        self.getOutput(0).setData(self.binary*255)


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-p", "--path", help="the path of the image folder", default=None)
    options, remain = parser.parse_args()
    exts = ["*.raw"]
    files = [] # display images from file
    if options.path != None:
        # grab files into a list
        for ext in exts:
            files += glob.glob(os.path.join(options.path,ext))   
    files.sort()
    # read in the background image
    bgImg = source.readImageFile(files[0])
    bgImg = (bgImg * 255.0/4095.0).astype(numpy.uint8)
    cv2.imshow("bgImg", bgImg)

    # read in all images
    fileStackReader = source.FileStackReader(files)
    
    # convert to 8 bit
    eightbitimages = AffineIntensity(fileStackReader.getOutput())
    eightbitimages.setScale(255.0/4095.0)
    
    # do binary segmentation
    binarySeg = BinarySegmentation(eightbitimages.getOutput(), bgImg)
    display1 = Display(eightbitimages.getOutput(),"original")
    display2 = Display(binarySeg.getOutput(),"binary segmentation")
    key = cv2.waitKey(10)
    while key != 27:
      fileStackReader.increment()
      #print fileStackReader.getFrameName()
      display1.update()
      display2.update()
      cv2.waitKey(10)
