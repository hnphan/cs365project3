/**
 * Declares a set of functions to simplify interaction with libdc1394,
 * a library for image capture from firewire cameras.
 */

#ifndef FIREWIREVIDEO_H
#define FIREWIREVIDEO_H

#include <stdlib.h>
#include <inttypes.h>
#include <dc1394/dc1394.h>

typedef unsigned char uint8;
typedef unsigned short int uint16;
typedef unsigned int uint32;
typedef unsigned int uint;

/**
 * Data structure representing a firewire device, camera, and acquired
 * video frames.
 */
typedef struct {
    dc1394_t *fdevice;
    dc1394camera_t *camera;
    dc1394video_frame_t *rawframe;
    dc1394video_frame_t *rgbframe;
    uint mode;
    uint width;
    uint height;
    uint buffers;
    float framerate;
    uint demosaic;
} FV_FirewireCamera;

typedef enum
{
	DEMOSAIC_RGB = 1,
	SWAP_BYTES = 2,
	SHIFT_BITS4 = 4
} FV_RawProcess;

/**
 * Function pointer for allocating an RGB image.  The first parameter is a
 * pointer to 3 integers specifying the height, width, and number of channels
 * to allocate.  The second parameter is an integer specifying the number of
 * bytes per pixel to allocate.
 */
typedef void* (*image_alloc_t)(int*, int);

// Prototypes
/**
 * Open a firewire camera for video capture.  The video mode flag
 * specifies the frame size and color format.  The camera library will
 * use the given number of buffers in its DMA circular buffer.  Returns
 * a pointer to a camera structure that is used for all other library
 * calls.
 */
FV_FirewireCamera* FV_openVideoDevice(int deviceNumber, int isoSpeed, int buffers);

/**
 * Sets the camera video mode to a standard (VESA) format.
 */
int FV_setVideoMode(FV_FirewireCamera* fcd, int videoMode, int framerate);

/**
 * Sets the camera video mode to a Format7 flexible format.
 */
int FV_setVideoModeF7(FV_FirewireCamera* fcd, int videoMode, int colorCoding, uint width, uint height, uint left, uint top, float targetFPS, int demosaic);

/**
 * Starts the transmission of camera frames from the camera.
 */
int FV_startTransmission(FV_FirewireCamera* fcd);

/**
 * Stops the transmission of video frames from the camera.  Optionally
 * flushes the image ring buffer.
 */
int FV_stopTransmission(FV_FirewireCamera* fcd, int flush);

/**
 * Acquire an image frame and store its data in the memory block allocated by
 * the allocator function.
 */
void* FV_acquireFrame(FV_FirewireCamera* fcd, image_alloc_t allocator);

/**
 * Acquire a single frame from the camera.  This is used when one needs
 * control over the acquisition of a single frame from the camera.
 * Video transmission should be stopped before using this method:
 * FV_stopTransmission(fcd, 1);
 */
void* FV_oneShot(FV_FirewireCamera* fcd, image_alloc_t allocator);

/**
 * Flush the DMA ring buffer to ensure the next frame acquired is current.
 */
int FV_flushRingBuffer(FV_FirewireCamera* fcd);

/**
 * Set a feature register value.  Setting a feature switches it into manual
 * mode and disables absolute value control.  Setting shutter or gain disables
 * auto exposure.
 */
int FV_setFeatureRegister(FV_FirewireCamera* fcd, int featureId, uint value);

/**
 * Get a feature register value.
 */
int FV_getFeatureRegister(FV_FirewireCamera* fcd, int featureId, uint* value);

/**
 * Set a feature absolute value.  Setting a feature switches it into manual
 * mode and enables absolute control.  Setting shutter or gain disables
 * auto exposure.
 */
int FV_setFeatureAbsolute(FV_FirewireCamera* fcd, int featureId, float value);

/**
 * Get a feature absolute value.
 */
int FV_getFeatureAbsolute(FV_FirewireCamera* fcd, int featureId, float* value);

/**
 * Sets the white balance parameters for the camera.
 */
int FV_setWhiteBalanceRegister(FV_FirewireCamera *fcd, uint blueValue, uint redValue);

/**
 * Gets the white balance parameters for the camera.
 */
int FV_getWhiteBalanceRegister(FV_FirewireCamera *fcd, uint* blueValue, uint* redValue);

/**
 * Turn a feature on or off.  0 for off, 1 (non-zero) for on.
 */
int FV_setFeaturePower(FV_FirewireCamera* fcd, int featureId, int power);

/**
 * Get the power status of a feature.
 */
int FV_getFeaturePower(FV_FirewireCamera* fcd, int featureId, int* power);

/**
 * Sets the operation mode (manual, auto, one push) for a feature.
 */
int FV_setFeatureMode(FV_FirewireCamera* fcd, int featureId, int mode);

/**
 * Gets the operation mode for a feature.
 */
int FV_getFeatureMode(FV_FirewireCamera* fcd, int featureId, int* mode);

/**
 * Gets the color filter id for a camera, dependent on the camera being
 * in a FORMAT7 video mode with raw acquisition.
 */
int FV_getColorFilter(FV_FirewireCamera* fcd, int* filter);

/**
 * Enable or disable auto exposure.  Auto exposure is controlled by
 * exposure value (EV), shutter, and gain.  This function sets all those
 * features to auto mode if enable is non-zero, or to manual if enable
 * is zero.
 */
int FV_setAutoExposure(FV_FirewireCamera* fcd, int enable);

/**
 * Load current values, minimum, maximum, and valid flags for all libdc1394 features.
 * Register and absolute values are determined for each feature.  The feature values
 * are written into the uint (register) and float (absolute) arrays provided.
 *
 * The register array should hold NUM_FEATURES * 5 elements, where NUM_FEATURES
 * is (DC1394_FEATURE_MAX-DC1394_FEATURE_MIN+1).  For each array, the elements
 * returned are: value, min, max, available, power
 *
 * The absolute array should hold NUM_FEATURES * 5 elements.  The elements
 * returned are: value, min, max, available
 */
int FV_getAllFeatureInfo(FV_FirewireCamera* fcd, uint* registerValues, float* absoluteValues);

/**
 * Print camera features and current values using libdc1394 function.
 */
void FV_printFeatures(FV_FirewireCamera* fcd);

/**
 * Close the firewire camera and free up all resources.
 */
void FV_closeVideoDevice(FV_FirewireCamera* fcd);

void* allocateFrame(int* shape, int bytesPerPixel);

/**
 * Prints the binary representation of a block of data, from the beginning
 * to the specified size, in bytes.
 */
void printBinary(void* data, uint size);

/**
 * Prints dc1394 image frame information, including min/max intensity
 * values.
 */
void imageInfo(dc1394video_frame_t* image);
void swapByteOrder(dc1394video_frame_t* image);
void shiftBits(dc1394video_frame_t* image, uint bits);

/**
 * Compute the Format7 framerate from camera parameters and packet size in bytes.
 */
float computeFramerate(FV_FirewireCamera* fcd, uint colorCoding, uint packetBytes);

/**
 * Compute the Format7 packet size in bytes from camera parameters
 * and target framerate.
 */
uint computePacketSize(FV_FirewireCamera* fcd, uint colorCoding, float framerate);

#endif
