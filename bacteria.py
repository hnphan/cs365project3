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

import avgimage

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
        #print "output: ", output[694,713]

        tempBinary = numpy.zeros(output.shape)
        tempBinary[output < 30] = 1
        tempBinary[output >= 30] = 0

        tempBinary = ndimage.grey_erosion(tempBinary, size = (3,3))
        tempBinary = ndimage.grey_erosion(tempBinary, size = (3,3))
        tempBinary = ndimage.grey_dilation(tempBinary, size = (3,3))

        self.binary = numpy.logical_or(self.binary, tempBinary).astype(numpy.uint8)
        self.getOutput(0).setData(self.binary*255)

class FilterFlatField(pipeline.ProcessObject):
    """
        Applies the flat field image to the input stream
    """
    
    def __init__(self, input=None, ff_image=None):
        pipeline.ProcessObject.__init__(self, input)
        if ff_image != None:
            self.setFFImage(ff_image)
        
    def setFFImage(self, ff_image):
        self.ff_image = ff_image
        self.modified()
    
    def generateData(self):
        inpt = self.getInput(0).getData()
        output = inpt*self.ff_image
        self.getOutput(0).setData(output)


class Cropper(pipeline.ProcessObject):
    """
        Crops the input stream to the given dimensions (see crop_np_image)
    """
    
    def __init__(self, input = None, crop_dimensions = None):
        pipeline.ProcessObject.__init__(self, input)
        if crop_dimensions != None:
            self.setCropDimensions(crop_dimensions)
        
    def setCropDimensions(self, crop_dimensions):
        self.crop_dimensions = crop_dimensions
        self.modified()

    def generateData(self):
        inpt = self.getInput(0).getData()
        cropped_image = crop_np_image(inpt, self.crop_dimensions)
        self.getOutput(0).setData(cropped_image)

def crop_np_image( input_image, crop_dimensions ):
    """
        Crops an input numpy image to the given dimensions

        crop_dimensions: [(ystart,yend), (xstart,xend)]
    """
    (ystart, yend), (xstart,xend) = crop_dimensions
    output = input_image[ystart : yend, xstart : xend]
    return output

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-p", "--path", help="the path of the image folder", default=None)
    options, remain = parser.parse_args()
    exts = ["bact*.raw"]
    flat_exts = ["flat*.raw"]
    files = [] # display images from file
    flat_files = [] # flat-field images
    if options.path != None:
        # grab files into a list
        for ext in exts:
            files += glob.glob(os.path.join(options.path,ext))   
        for ext in flat_exts:
            flat_files += glob.glob(os.path.join(options.path, ext))

    files.sort()
    flat_files.sort()

    # Specify dimensions all images are to be cropped to
    # Crop to  [(ymin, ymax), (xmin, xmax)]
    dimensions = [ (100, 920), (45, 1315) ]

    # Read in the background image
    bgImg = source.readImageFile(files[0])
    bgImg = (bgImg * 255.0/4095.0).astype(numpy.uint8)
    bgImg = crop_np_image( bgImg, dimensions)
    #cv2.imshow("bgImg", bgImg)

    # Read in all images, crop them
    fileStackReader = source.FileStackReader(files)
    cropped_images = Cropper(fileStackReader.getOutput(), dimensions)
    
    # convert to 8 bit
    eightbitimages = AffineIntensity(cropped_images.getOutput())
    eightbitimages.setScale(255.0/4095.0)
    
    # Obtain the flat-field image
    savename = "flat_field.npy"
    if not os.path.isfile(savename):
        flat_images = ( source.readImageFile(fn) for fn in flat_files )
        cropped_flat_images = (crop_np_image(img, dimensions) for img in flat_images)
        flat_field = avgimage.get_flat_field(cropped_flat_images, len(flat_files))
        numpy.save(savename, flat_field)
    else:
        print "Loading flat field '%s'" % savename
        flat_field = numpy.load(savename)
        (ystart, yend), (xstart,xend) = dimensions
        cropped_shape = ( yend - ystart, xend-xstart )
        assert flat_field.shape == cropped_shape, ( "Flat field size differs "
            "from image sizes. Try deleting file.")
    #cv2.imshow("Flat field image", flat_field)

    # Apply flat-fielding to the images
    corrected_images = FilterFlatField( eightbitimages.getOutput(), flat_field)

    # Do binary segmentation
    binarySeg = BinarySegmentation(corrected_images.getOutput(), bgImg)

    # Display images
    display1 = Display(eightbitimages.getOutput(),"Original Image")
    display2 = Display(corrected_images.getOutput(),"Flat-fielded Image")
    display3 = Display(binarySeg.getOutput(),"Binary Segmentation")
    key = None
    while key != 27:
      fileStackReader.increment()
      print fileStackReader.getFrameName()
      display1.update()
      display2.update()
      display3.update()
      cv2.waitKey(10)
