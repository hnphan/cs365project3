#!/usr/bin/env python

# David Cain
# Hieu Phan
# Justin Sperry
# 2012-03-30
# CS365, Brian Eastwood

'''
Created on Feb 20, 2012

@author: bseastwo
'''

import math
import os

from scipy import ndimage
import cv
import cv2
import glob
import numpy
import optparse
import pylab

import avgimage
import pipeline
import source

class Dishes:
    """
        Mimics an enumerated type to store the different Petri dishes
        with bacteria colonies.
    """
    upper_middle, upper_right, lower_left, lower_middle = range(4)


class Display(pipeline.ProcessObject):
    """
        Pipeline object to display the numpy image in a CV window
    """
    
    def __init__(self, input=None, name="pipeline"):
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


class ImageCorrect(pipeline.ProcessObject):
    """
        Corrects the image by cropping a relevant region, converting the dtype.
    """
    def __init__(self, input=None, crop_dimensions=None, dtype=numpy.float64):
        pipeline.ProcessObject.__init__(self, input)
        self.dtype = dtype
        if crop_dimensions != None:
            self.setCropDimensions(crop_dimensions)
        
    def setCropDimensions(self, crop_dimensions):
        """
            Sets the crop dimensions to apply when generating the image.

            crop_dimensions: [(ystart,yend), (xstart,xend)]
        """
        self.crop_dimensions = crop_dimensions
        self.modified()

    def generateData(self):
        """
            Crop the image to the desired dimensions and change the type
        """
        inpt = self.getInput(0).getData()
        cropped_image = crop_image(inpt, self.crop_dimensions)

        scaled_cropped = scale_image(cropped_image, self.dtype)
        self.getOutput(0).setData(scaled_cropped)

def crop_image( input_image, crop_dimensions ):
    """
        Crops an input numpy image to the given dimensions.

        input_image: a numpy image
        crop_dimensions: [(ystart,yend), (xstart,xend)]
    """
    (ystart, yend), (xstart,xend) = crop_dimensions
    output = input_image[ystart : yend, xstart : xend]
    return output

def scale_image( input_image, dtype):
    """
        Scale the input image to its max value, assign the given type

        input_image: a numpy image
        dtype: a numpy type (e.g. numpy.float64)
    """
    input_image *= (255.0 / input_image.max())
    input_image = input_image.astype(dtype)
    return input_image


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


class BinarySegmentation(pipeline.ProcessObject):
    """
        Segments the bacteria colonies in the images.
    """
    def __init__(self, input=None, bgImg=None, std_bgImg=None, alpha=None):
        pipeline.ProcessObject.__init__(self, input)
        self.bgImg = bgImg
        self.binary = numpy.zeros(bgImg.shape)
        self.std_bgImg = std_bgImg
        self.alpha = alpha
    
    def generateData(self):
        """
            Perform background subtraction on the image, segment the
            bacteria colonies (foreground) from the background data.
        """
        input = self.getInput(0).getData()
        #background subtraction
        output = (input.astype(numpy.float) - self.bgImg.astype(numpy.float))
        
        #threshold = self.alpha * std_bgImg
        #print threshold[100:200,100:200]
        tempBinary = numpy.zeros(output.shape)
        tempBinary[output < -5] = 1
        
        tempBinary = ndimage.morphology.binary_opening(tempBinary, iterations = 5)
        
        self.binary = numpy.logical_or(self.binary, tempBinary).astype(numpy.uint8)
        self.getOutput(0).setData(self.binary*255)
        
        
class Perimeter(pipeline.ProcessObject):
    """
        Obtain the perimeter of the bacteria colonies, to be used in
        calculating region properties data
    """
    def __init__(self, input = None, orgImg = None):
        pipeline.ProcessObject.__init__(self, input)
        self.setOutput(input, 1)
        
    def generateData(self):
        input = self.getInput(0).getData()/255
        fill = ndimage.grey_erosion(input, size = (3,3))
        
        output = input - fill
        self.getOutput(0).setData(output*255)
        self.getOutput(1).setData(input*255)        


class RegionProperties(pipeline.ProcessObject):
    """
        Calculates area, center of mass, and circularity of the bacteria
        colonies.
    """  
    
    #Constructor, regions and a mask are optional
    def __init__(self, input=None, input1=None, num_data_slides=140):
        pipeline.ProcessObject.__init__(self, input)
        self.count = 0
        self.setInput(input1, 1)
        self.num_data_slides = num_data_slides
        self.store = numpy.zeros((4, 2, self.num_data_slides))
    
    def generateData(self):
        """
            Iterate through the 140 data slides, obtaining region properties and
            generate display windows of the information when completed.
        """
        
        if self.count < self.num_data_slides: # Generate data
            #grabs input and converts it to binary
            input = self.getInput(1).getData()/255
            perimeters = self.getInput(0).getData()/255
            
            #dividers to break the image into 6 evenly sized boxes
            one_third = input.shape[1]/3
            two_thirds = 2*one_third
            half = input.shape[0]/2
            
            #label and convert to sequential indices
            labels, count = ndimage.label(input)
            l = numpy.unique(labels)
            for each in range(1,l.size):
                labels[labels == l[each]] = each
            
            #grab slices from these labels for use in perimeters
            slices = ndimage.find_objects(labels)
            
            print "There are %s regions" % (count)
            # loop through each identified region
            for i in range(1,numpy.unique(labels).size):
            
                # calculate the center of mass and area of the region
                com = ndimage.measurements.center_of_mass(input,labels,i)
                area = numpy.count_nonzero(input[slices[i-1]])
                
                # calculates circularity
                p = numpy.count_nonzero(perimeters[slices[i-1]])
    
                # checks to make sure perimeter is not zero
                if (p != 0):
                    abscirc = ((p*p)/area)
                    circularity = (4*math.pi)/abscirc
                else:
                    circularity = 0
    
                metrics = numpy.array([area, circularity])
                
                print "Colony %s at %s has an area of %s" % (i, com, area)
                #print "Circularity : %s " %(circularity)
                
                #decides which bin to store area and circularity in
                if com[0] < half:
                    if com[1] < two_thirds:
                        box = Dishes.upper_middle
                    else:
                        box = Dishes.upper_right
                else:
                    if com[1] < one_third:
                        box = Dishes.lower_left
                    elif com[1] < two_thirds:
                        box = Dishes.lower_middle
                    else:
                        box = 5
                
                if box < 5:
                    self.store[box,:, self.count] = metrics
                
            self.count += 1      
        
        # When finished iterating through the slides, generate and plot data
        elif self.count == self.num_data_slides: 
            time = range(60,200)
            pylab.subplot(211)

            # Plot area against time for each dish region
            area_um = self.store[Dishes.upper_middle,0,:]
            area_ur = self.store[Dishes.upper_right,0,:]
            area_ll = self.store[Dishes.lower_left,0,:]
            area_lm = self.store[Dishes.lower_middle,0,:]
            
            p1, = pylab.plot(time,area_um,'r')
            p2, = pylab.plot(time , area_ur, 'g')
            p3, = pylab.plot(time, area_ll, 'b')
            p4, = pylab.plot(time, area_lm, 'k')
            pylab.legend([p1,p2,p3,p4], ["Upper Middle", "Upper Right", "Lower Left", "Lower Middle"], loc=1)

            pylab.title('Area Against Time')
            pylab.xlabel('Time (Frames)')
            pylab.ylabel('Area (Pixels)')
            
            # Plot circularity data
            pylab.subplot(212)
            circularity_um = self.store[Dishes.upper_middle,1,:]
            circularity_ur = self.store[Dishes.upper_right,1,:]
            circularity_ll = self.store[Dishes.lower_left,1,:]
            circularity_lm = self.store[Dishes.lower_middle,1,:]
            
            p1, = pylab.plot(time,circularity_um,'r')
            p2, = pylab.plot(time , circularity_ur, 'g')
            p3, = pylab.plot(time, circularity_ll, 'b')
            p4, = pylab.plot(time, circularity_lm, 'k')

            #pylab.legend([p1,p2,p3,p4], ["Upper Middle", "Upper Right", "Lower Left", "Lower Middle"])
            #pylab.plot(time,circularity_um,'r', time , circularity_ur, 'g', 
            #        time, circularity_ll, 'b',time, circularity_lm, 'k')

            pylab.title('Circularity Over Time')
            pylab.xlabel('Time (Frames)')
            pylab.ylabel('circularity (Pixels)')
            
            pylab.show()
        
        else:
            pass
        
        self.getOutput(0).setData(input)    

        
class ColonyVisualize(pipeline.ProcessObject):
	"""
        Visualize the detected colony region by drawing red borders
	"""
	def __init__(self, input = None, orgImg = None):
		pipeline.ProcessObject.__init__(self, input)
		self.setInput(orgImg,1)
	
	def generateData(self):
		input = self.getInput(0).getData()
		orgImg = self.getInput(1).getData()
		output = numpy.zeros((input.shape[0],input.shape[1],3))
		output[...,0] = orgImg
		output[...,1] = orgImg
		output[...,2] = orgImg
		output[...,0][input>0] = 150
		self.getOutput(0).setData(output)


def main():
    """
        Given a path to the image folder, perform all the operations on
        the bacteria colony data:
           - flat fielding
           - binary segmentation of bacteria
           - calculating and displaying properties of the colonies 
           - obtaining and displaying the perimeter of bacteria colonies
    """

    # Take the path to the image folder, obtain lists of the filenames
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
    
    # Obtain the flat-field image. If a file exists by the given
    # savename, load the data in. Otherwise, generate the image from the
    # flat field frames.
    savename = "flat_field.npy"
    if not os.path.isfile(savename):
        flat_images = ( source.readImageFile(fn) for fn in flat_files )
        cropped_flat_images = (crop_image(img, dimensions) for img in flat_images)
        flat_field = avgimage.get_flat_field(cropped_flat_images, len(flat_files))
        numpy.save(savename, flat_field)
    else:
        print "Loading flat field '%s'" % savename
        flat_field = numpy.load(savename)
        (ystart, yend), (xstart,xend) = dimensions
        cropped_shape = ( yend - ystart, xend-xstart )
        assert flat_field.shape == cropped_shape, ( "Flat field size differs "
            "from image sizes. Try deleting file.")

    # Read in the background image, flat-field correct
    # Generate the files if one does not exist, otherwise load from .npy's
    std_bg_savename = "std_bgImg.npy"
    mean_bg_savename = "mean_bgImg.npy"

    if not (os.path.isfile(std_bg_savename) and os.path.isfile(mean_bg_savename)):
        print "Generating '%s' and '%s'" % (std_bg_savename, mean_bg_savename)

        num_bg_images = 60
        bgImg = numpy.zeros((1036, 1388, num_bg_images))
        for i in range(num_bg_images):
            cur_bgImg = source.readImageFile(files[i])
            bgImg[:,:,i] = cur_bgImg

        std_bgImg = numpy.std(bgImg,axis=2)
        std_bgImg = crop_image(std_bgImg, dimensions)
        std_bgImg = scale_image(std_bgImg, numpy.float64)
        
        mean_bgImg = numpy.mean(bgImg,axis=2)
        mean_bgImg = crop_image(mean_bgImg, dimensions)
        mean_bgImg = scale_image(mean_bgImg, numpy.float64)
        mean_bgImg *= flat_field

        print "Saving to file..."
        numpy.save(std_bg_savename, std_bgImg)
        numpy.save(mean_bg_savename, mean_bgImg)
    else:
        print("Loading saved backround images '%s' and '%s'" %
            (std_bg_savename, mean_bg_savename))
        std_bgImg = numpy.load(std_bg_savename)
        mean_bgImg = numpy.load(mean_bg_savename)

    # Read in all images, crop them
    fileStackReader = source.FileStackReader(files[60:])
    cropped_images = ImageCorrect(fileStackReader.getOutput(), dimensions)
    
    # Apply flat-fielding to the images
    corrected_images = FilterFlatField( cropped_images.getOutput(), flat_field)

    # Do binary segmentation
    binarySeg = BinarySegmentation(corrected_images.getOutput(), mean_bgImg, std_bgImg, 5)

    # Get perimeter
    perimeter = Perimeter(binarySeg.getOutput())
    
    # Calculate region properties
    regProperties = RegionProperties(perimeter.getOutput(0), perimeter.getOutput(1))
    
    # Visualize colonies
    visualColonies = ColonyVisualize(perimeter.getOutput(0), corrected_images.getOutput())
    
    # Display generated images
    cropped_display = Display(cropped_images.getOutput(),"Original Image")
    flat_display = Display(corrected_images.getOutput(),"Flat-fielded Image")
    seg_display = Display(binarySeg.getOutput(),"Binary Segmentation")
    colony_display = Display(visualColonies.getOutput(), "Colony Visualization")

    display_windows = [cropped_display, flat_display, seg_display,
            colony_display]
    
    # Iterate through each frame, displaying different windows for each
    # part of the process.
    key = None
    while key != 27:
      fileStackReader.increment()
      print fileStackReader.getFrameName()
      for window in display_windows:
          window.update()
      regProperties.update()
      cv2.waitKey(10)

if __name__ == "__main__":
    main()
