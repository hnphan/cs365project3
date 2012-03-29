/*
 * Functions to interact with a dc1394 firewire camera.
 * Adapted from code by Damien Douxchamps <ddouxchamps@users.sf.net>
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <unistd.h>
#include <dc1394/dc1394.h>
#include <firewireVideo.h>

#define FV_VERBOSE 1

/**
 * Logging function; when FV_VERBOSE is 0, the precompiler takes the
 * the logging function calls away.
 */
#if FV_VERBOSE
#define verbose(...) printf(__VA_ARGS__)
#else
#define verbose(...)
#endif

/**
 * Error handling macro borrowed from libdc1394, but does not return.
 */
#define handleError(err, message)                         \
  do {                                                    \
    if ((err>0)||(err<=-DC1394_ERROR_NUM))                \
      err=DC1394_INVALID_ERROR_CODE;                      \
                                                          \
    if (err!=DC1394_SUCCESS) {                            \
      dc1394_log_error("%s: in %s (%s, line %d): %s\n",   \
      dc1394_error_get_string(err),                       \
          __FUNCTION__, __FILE__, __LINE__, message);     \
    }                                                     \
  } while (0);

/*
 * Opens a firewire camera and returns a pointer to a camera structure.
 * If logging is enabled, this function prints out information about all
 * firewire cameras attached to the computer and the video modes supported
 * by the selected camera.
 *
 * Input:
 * 	deviceNumber - the index of the firewire camera to attach to.  Note
 * 		that if the 1394 host card is reset, the camera indices can change
 * 	isoSpeed - the dc1394speed_t flag that specifies the transmission rate
 * 		to use for the camera.  There is no known way to determine if the
 * 		1394 host card supports 1394b mode (800 Mbps), so this needs to
 * 		be specified programmatically.
 * 	buffers - determines the number of ring image buffers configured for
 * 		the camera.  Although this is not used until a video mode is
 * 		selected, it is unlikely to change for different video modes.
 */
FV_FirewireCamera* FV_openVideoDevice(
		int deviceNumber,
		int isoSpeed,
		int buffers)
{
	verbose("FV_FirewireCamera: %2d\n", deviceNumber);
	FV_FirewireCamera* fcd;
	dc1394error_t err = DC1394_SUCCESS;
	int i;

	// allocate and initialize the camera structure
	fcd = malloc(sizeof(FV_FirewireCamera));
	verbose("Camera pointer: %p\n", fcd);
	fcd->fdevice = NULL;
	fcd->camera = NULL;
	fcd->rawframe = NULL;
	fcd->rgbframe = NULL;
	fcd->mode = 0;
	fcd->width = 0;
	fcd->height = 0;
	fcd->framerate = 0.0;
	fcd->buffers = buffers;
	fcd->demosaic = 1;

	// find cameras attached to 1394 device
	fcd->fdevice = dc1394_new();
	dc1394camera_list_t *list; // list of camera identifiers
	err = dc1394_camera_enumerate(fcd->fdevice, &list);
	handleError(err, "Failed to enumerate cameras.");

	if (err == DC1394_SUCCESS)
	{
		// check if there are any cameras attached
		if (list->num == 0)
		{
			err = DC1394_NOT_A_CAMERA;
			handleError(err, "No cameras found");
			printf("No cameras found on device; acquisition functions will return test images.\n");
			dc1394_camera_free_list(list);
		}
		else
		{
#if FV_VERBOSE
			// print out the list of cameras
			printf("\n------ Camera listing ------\n");
			for (i = 0; i < list->num; i++)
			{
				printf("%2d: guid %llX unit %hX\n", i, list->ids[i].guid, list->ids[i].unit);
			}
			printf(" Using: %d\n", deviceNumber);
			printf("----------------------------\n\n");
#endif
		}
	}

	if (err == DC1394_SUCCESS)
	{
		// open the requested camera device
		fcd->camera = dc1394_camera_new(fcd->fdevice, list->ids[deviceNumber].guid);
		dc1394_camera_free_list(list);
		if (!fcd->camera)
		{
			err = DC1394_CAMERA_NOT_INITIALIZED;
			handleError(err, "Failed to initialize camera");
		}
	}

	if (err == DC1394_SUCCESS)
	{
		// set the 1394 bus transmission speed
		// (iso stands for isochronous, not at all like film speed)
		// all cameras initialize at 400 Mbps (1394a)
		// todo: Is there a way to determine what mode the 1394 host
		// supports?  A camera that supports 1394b can still be plugged
		// into a 1394a port, but this does not change the value of
		// camera->bmode_capable.  Another way to determine bus speed
		// would be handy.

		if (isoSpeed < DC1394_ISO_SPEED_100 || isoSpeed > DC1394_ISO_SPEED_3200)
			isoSpeed = DC1394_ISO_SPEED_400;

//		if (fcd->camera->bmode_capable)
		if (isoSpeed >= DC1394_ISO_SPEED_800)
		{
			dc1394_video_set_operation_mode(fcd->camera, DC1394_OPERATION_MODE_1394B);
			err = dc1394_video_set_iso_speed(fcd->camera, isoSpeed);
		}
		else
		{
			dc1394_video_set_operation_mode(fcd->camera, DC1394_OPERATION_MODE_LEGACY);
			err = dc1394_video_set_iso_speed(fcd->camera, isoSpeed);
		}
		handleError(err, "Failed to set ISO speed");
	}

#if FV_VERBOSE
	if (err == DC1394_SUCCESS)
	{
		// camera information
		dc1394_camera_print_info(fcd->camera, stdout);
		dc1394speed_t iso;
		dc1394_video_get_iso_speed(fcd->camera, &iso);
		char* fwmode = iso > 2 ? "1394b" : "1394a";
		printf("Using ISO speed: %d (%s)\n", iso, fwmode);

		// video modes
		dc1394video_modes_t modes;
		dc1394_video_get_supported_modes(fcd->camera, &modes);
		dc1394color_coding_t color;
		unsigned int width, height, bpp;
		char* format;
		printf("\n------ Camera video modes ------\n");
		for (i = 0; i < modes.num; i++)
		{
			dc1394_get_image_size_from_video_mode(fcd->camera, modes.modes[i], &width, &height);
			dc1394_get_color_coding_from_video_mode(fcd->camera, modes.modes[i], &color);
			dc1394_get_color_coding_bit_size(color, &bpp);
			format = modes.modes[i] < DC1394_VIDEO_MODE_FORMAT7_0 ? "Standard" : "Format7";
			printf("%2d: %2d  (%4d x%4d, %2d bpp, %s)\n", i, (int) modes.modes[i], width, height, bpp, format);
		}
		printf("--------------------------------\n");
	}
#endif

	// return the camera data structure if all went well
	if (err != DC1394_SUCCESS)
	{
		FV_closeVideoDevice(fcd);
		return (NULL);
	}

	return fcd;
}

int FV_setVideoMode(FV_FirewireCamera* fcd, int videoMode, int framerate)
{
	verbose("FV_setVideoMode: mode %2d, rate %2d\n", videoMode, framerate);
	if (!fcd) return DC1394_SUCCESS;

	dc1394error_t err = DC1394_SUCCESS;
	verbose("Setting camera mode\n");
	fcd->mode = videoMode;
	verbose("Demosaic\n");
	fcd->demosaic = DEMOSAIC_RGB;

	// stop video transmission
	verbose("Stopping transmission...\n");
	dc1394_video_set_transmission(fcd->camera, DC1394_OFF);
	dc1394_capture_stop(fcd->camera);

	// set video mode
	verbose("Setting video mode...\n");
	err = dc1394_video_set_mode(fcd->camera, fcd->mode);
	handleError(err, "Failed to set video mode.");

	if (err == DC1394_SUCCESS)
	{
	    verbose("Finding frame properties...\n");

		// determine frame properties
		dc1394_get_image_size_from_video_mode(
			fcd->camera, fcd->mode, &(fcd->width), &(fcd->height));

		// determine framerates
		dc1394framerates_t rates;
		dc1394_video_get_supported_framerates(fcd->camera, fcd->mode, &rates);

		// use maximum framerate if not specified
		if (framerate < DC1394_FRAMERATE_MIN)
		{
			framerate = rates.framerates[rates.num-1];
		}
		err = dc1394_video_set_framerate(fcd->camera, framerate);
		handleError(err, "Failed to set framerate.");
		dc1394_framerate_as_float(framerate, &(fcd->framerate));

#if FV_VERBOSE
		int i;
		float fps;
		printf("\n------ Framerates for video mode %2d ------\n", fcd->mode);
		for (i = 0; i < rates.num; i++)
		{
			dc1394_framerate_as_float(rates.framerates[i], &fps);
			printf("%2d: %2d (%5.2f fps)\n", i, rates.framerates[i], fps);
		}
		printf(" Using: %d (%5.2f fps)\n", framerate, fcd->framerate);
		printf("------------------------------------------\n");
#endif
	}

	// setup data capture from the camera
	if (err == DC1394_SUCCESS)
	{
		// the number of images in the ring buffer should be 4-10
		err = dc1394_capture_setup(fcd->camera, fcd->buffers, DC1394_CAPTURE_FLAGS_DEFAULT);
		handleError(err, "Failed to setup data capture.");
	}

	// setup the RGB frame where image data is transferred to
	if (err == DC1394_SUCCESS)
	{
		// clean up if we have allocated before
		if (fcd->rgbframe && fcd->rgbframe->image)
			free(fcd->rgbframe->image);
		if (fcd->rgbframe)
			free(fcd->rgbframe);

		// allocate space for the rgb frame
		fcd->rgbframe = calloc(1, sizeof(dc1394video_frame_t));

		// configure frame and allocate memory
		fcd->rgbframe->size[0] = fcd->width;
		fcd->rgbframe->size[1] = fcd->height;
		fcd->rgbframe->color_coding = DC1394_COLOR_CODING_RGB8;
		int shape[3] = {fcd->height, fcd->width, 3};
		fcd->rgbframe->image = (uint8*) allocateFrame(shape, sizeof(uint8));
	}

	return err;
}

int FV_setVideoModeF7(FV_FirewireCamera* fcd, int videoMode, int colorCoding, uint width, uint height, uint posX, uint posY, float targetFPS, int demosaic)
{
	verbose("FV_setVideoModeF7: mode %2d, color %2d, fps %5.3f\n", videoMode, colorCoding, targetFPS);
	if (!fcd) return DC1394_SUCCESS;

	dc1394error_t err = DC1394_SUCCESS;
	fcd->mode = videoMode;

	// stop video transmission
	dc1394_video_set_transmission(fcd->camera, DC1394_OFF);
	dc1394_capture_stop(fcd->camera);

	// set video mode
	err = dc1394_video_set_mode(fcd->camera, fcd->mode);
	handleError(err, "Failed to set video mode.");

	if (err == DC1394_SUCCESS)
	{
		// determine format7 image size properties
		uint maxW, maxH, unitW, unitH, unitX, unitY;
		dc1394_format7_get_max_image_size(fcd->camera, fcd->mode, &maxW, &maxH);
		dc1394_format7_get_unit_size(fcd->camera, fcd->mode, &unitW, &unitH);
		dc1394_format7_get_unit_position(fcd->camera, fcd->mode, &unitX, &unitY);

		// validate requested ROI
		verbose("ROI request: %d x %d + (%d, %d)\n", width, height, posX, posY);

		// size cannot exceed maximum
		if (width == 0 || width > maxW)
			width = maxW;
		if (height == 0 || height > maxH)
			height = maxH;

		// size cannot be less than the unit size
		if (width < unitW)
			width = unitW;
		if (height < unitH)
			height = unitH;

		// size must be a multiple of unit size
		width -= width % unitW;
		height -= height % unitH;

		// position cannot place edge outside sensor
		if (posX + width > maxW)
			posX = maxW - width;
		if (posY + height > maxH)
			posY = maxH - height;

		// position must be a multiple of unit position
		posX -= posX % unitX;
		posY -= posY % unitY;

		verbose("ROI valid:   %d x %d + (%d, %d)\n", width, height, posX, posY);
		fcd->width = width;
		fcd->height = height;

		// setup format 7 ROI
		dc1394_video_set_mode(fcd->camera, fcd->mode);
		dc1394_format7_set_color_coding(fcd->camera, fcd->mode, colorCoding);
		dc1394_format7_set_image_size(fcd->camera, fcd->mode, fcd->width, fcd->height);
		dc1394_format7_set_image_position(fcd->camera, fcd->mode, posX, posY);

		// the packet size (bytes per packet) controls the framerate for
		// format 7.  compute and set the packet size.
		uint compPacket, setPacket;
		compPacket = computePacketSize(fcd, colorCoding, targetFPS);
	 	err = dc1394_format7_set_packet_size(fcd->camera, fcd->mode, compPacket);
		handleError(err, "Failed to set packet size.");

		// determine the obtained framerate
		dc1394_format7_get_packet_size(fcd->camera, fcd->mode, &setPacket);
		fcd->framerate = computeFramerate(fcd, colorCoding, setPacket);
		verbose("framerate: %5.3f fps => %5.3f fps\n", targetFPS, fcd->framerate);

		// determine the color filter on the camera
		dc1394color_filter_t color;
		dc1394_format7_get_color_filter(fcd->camera, fcd->mode, &color);
		verbose("color filter: %u\n", color);
	}

	// setup data capture from the camera
	if (err == DC1394_SUCCESS)
	{
		// the number of images in the ring buffer should be 4-10
		err = dc1394_capture_setup(fcd->camera, fcd->buffers, DC1394_CAPTURE_FLAGS_DEFAULT);
		handleError(err, "Failed to setup data capture.");
	}

	// setup the RGB/raw frame where image data is transferred to
	if (err == DC1394_SUCCESS)
	{
		// clean up if we have allocated before
		if (fcd->rgbframe && fcd->rgbframe->image)
			free(fcd->rgbframe->image);
		if (fcd->rgbframe)
			free(fcd->rgbframe);

		// allocate space for the rgb frame
		fcd->rgbframe = calloc(1, sizeof(dc1394video_frame_t));
		fcd->rgbframe->size[0] = fcd->width;
		fcd->rgbframe->size[1] = fcd->height;

		// determine how to handle frame conversion
		int shape[3] = {fcd->height, fcd->width, 3};
		int bytesPerPixel = sizeof(uint8);
		if (colorCoding == DC1394_COLOR_CODING_RAW16 ||
			(colorCoding == DC1394_COLOR_CODING_RAW8 && !(demosaic & DEMOSAIC_RGB)))
		{
			// no conversion, store raw frames
			shape[2] = 1;
			fcd->demosaic = demosaic & (~DEMOSAIC_RGB);
			fcd->rgbframe->color_coding = colorCoding;

			if (colorCoding == DC1394_COLOR_CODING_RAW16)
			{
				bytesPerPixel = sizeof(uint16);
				// handle 16 bit data formats for a few different cameras
				if (fcd->camera->vendor_id == 0xb09d)	// point grey uses left-shifted data
					fcd->demosaic = fcd->demosaic | SWAP_BYTES | SHIFT_BITS4;
				else										// unibrain uses non-shifted data
					fcd->demosaic = fcd->demosaic | SWAP_BYTES;
			}
		}
		else
		{
			// convert to rgb8 format
			fcd->demosaic = DEMOSAIC_RGB	;
			fcd->rgbframe->color_coding = DC1394_COLOR_CODING_RGB8;
		}
		verbose("raw process: %u -> %u\n", demosaic, fcd->demosaic);
		fcd->rgbframe->data_depth = 8 * bytesPerPixel;
		fcd->rgbframe->image = (uint8*) allocateFrame(shape, bytesPerPixel);
	}

	return err;
}

int FV_startTransmission(FV_FirewireCamera* fcd)
{
	// start transmission
	verbose("FV_startTransmission\n");
	dc1394error_t err = DC1394_SUCCESS;
	if (fcd)
	{
		err = dc1394_video_set_transmission(fcd->camera, DC1394_ON);
		handleError(err, "Failed to start transmission.");
	}
	return err;
}

int FV_stopTransmission(FV_FirewireCamera* fcd, int flush)
{
	// stop transmission
	verbose("FV_stopTransmission\n");
	dc1394error_t err = DC1394_SUCCESS;

	if (fcd)
	{
		err = dc1394_video_set_transmission(fcd->camera, DC1394_OFF);
		handleError(err, "Failed to stop transmission.");

		if (flush)
		{
			// need to sleep one frame period to make sure we do not
			// miss the last frame coming from the camera
			long int usecs = (long int) ((1.0 / fcd->framerate) * 1e6);
			usleep(usecs);
			dc1394_video_set_one_shot(fcd->camera, DC1394_OFF);
			FV_flushRingBuffer(fcd);
		}
	}
	return err;
}

/**
 * Acquires a raw frame from the ring buffer and converts it to RGB8
 * format.  This conversion is done into a memory location created by
 * the allocator function if this is not NULL.  Otherwise, the conversion
 * is done into the reserved rgbframe on the fcd structure.
 *
 * This function returns a pointer to the RGB8 image data.
 *
 * The allocator function may be a Python call back. (This was the
 * intention, though surely other scenarios would work.)
 */
void* FV_acquireFrame(FV_FirewireCamera *fcd, image_alloc_t allocator)
{
	void* rgbTarget = NULL;

	if (fcd != NULL)
	{
		// determine the size of the output image buffer
		int imgSize[] = {fcd->height, fcd->width, 3};
		if (!(fcd->demosaic & DEMOSAIC_RGB))
			imgSize[2] = 1;
		int pixelSize = fcd->rgbframe->color_coding == DC1394_COLOR_CODING_RAW16 ?
				sizeof(unsigned short int) : sizeof(unsigned char);
		size_t imgBytes = pixelSize * imgSize[0] * imgSize[1] * imgSize[2];

		void* rgbFcd = fcd->rgbframe->image;
		if (allocator)
		{
			// if an allocator is provided, allocate space for rgb image
			rgbTarget = allocator(imgSize, pixelSize);
			fcd->rgbframe->image = rgbTarget;
		}
		else
		{
			// otherwise, use the space reserved when we set up fcd
			rgbTarget = rgbFcd;
		}

		// get the next filled frame from the ring buffer
		dc1394_capture_dequeue(fcd->camera, DC1394_CAPTURE_POLICY_WAIT, &(fcd->rawframe));
		if (fcd->rawframe != NULL)
		{
			if (fcd->demosaic & SWAP_BYTES)
				swapByteOrder(fcd->rawframe);	// big/little endian conversion
			if (fcd->demosaic & SHIFT_BITS4)
				shiftBits(fcd->rawframe, 4);		// bit shifting

			// convert captured frame to RGB frame
			if (fcd->rawframe->video_mode < DC1394_VIDEO_MODE_FORMAT7_0 ||
				fcd->rawframe->color_coding < DC1394_COLOR_CODING_RAW8)
			{
				// decode frame to RGB
//				verbose("converting to RGB8\n");
//				imageInfo(fcd->rawframe);
				dc1394_convert_frames(fcd->rawframe, fcd->rgbframe);
//				imageInfo(fcd->rgbframe);
			}
			else if (fcd->demosaic & DEMOSAIC_RGB)
			{
				// demosaic raw data and copy to newly allocated space
				// note we can only demosaic 8 bit images right now
//				verbose("demosaicing\n");
//				imageInfo(fcd->rawframe);
				dc1394_debayer_frames(fcd->rawframe, fcd->rgbframe, DC1394_BAYER_METHOD_BILINEAR);
//				imageInfo(fcd->rgbframe);
			}
			else
			{
				// only if using FORMAT 7, a RAW color coding, and demosaic is off
				// copy raw data to newly allocated space
//				verbose("raw data copy\n");
//				imageInfo(fcd->rawframe);
				memcpy(rgbTarget, fcd->rawframe->image, imgBytes);
//				imageInfo(fcd->rgbframe);
			}
		}

		// show what the first few bytes of image data look like
//		printf("raw frame image data:");
//		printBinary(fcd->rawframe->image, 32);
//		printf("rgb frame image data:");
//		printBinary(fcd->rgbframe->image, 32);

		// return raw frame to ring buffer asap
		dc1394_capture_enqueue(fcd->camera, fcd->rawframe);

		// restore fcd rgbframe pointer
		fcd->rgbframe->image = rgbFcd;

		// check if we might be missing frames
		if (fcd->rawframe->frames_behind >= (fcd->buffers - 2))
		{
			verbose("FV_acquireFrame: lagging %2d frames\n", fcd->rawframe->frames_behind);
			FV_flushRingBuffer(fcd);
		}
	}

	if (rgbTarget == NULL)
	{
		// unable to get a frame; make a test image
		int imgSize[] = {480, 640, 3};
		int pixels = imgSize[0] * imgSize[1];
		rgbTarget = allocator(imgSize, sizeof(unsigned char));
		char* rgbChar = (char*) rgbTarget;

		// fill image with some color gradients
		int r,c;
		for (r = 0; r < imgSize[0]; r++)
		{
			for (c = 0; c < imgSize[1]; c++)
			{
				*rgbChar++ = (char) (255.0 * (imgSize[1]-c) * (imgSize[0]-r) / pixels);
				*rgbChar++ = (char) (255.0 * c / imgSize[1]);
				*rgbChar++ = (char) (255.0 * r / imgSize[0]);
			}
		}
	}

	return rgbTarget;
}

void* FV_oneShot(FV_FirewireCamera *fcd, image_alloc_t allocator)
{
	if (fcd)
		dc1394_video_set_one_shot(fcd->camera, DC1394_ON);
	return FV_acquireFrame(fcd, allocator);
}

/**
 * Flushes the camera's ring buffer to ensure next frame is current.
 */
int FV_flushRingBuffer(FV_FirewireCamera *fcd)
{
	verbose("FV_flushRingBuffer\n");
	if (fcd)
	{
		// run polling (non-blocking) capture until no frame is returned
		dc1394_capture_dequeue(fcd->camera,
				DC1394_CAPTURE_POLICY_POLL,
				&(fcd->rawframe));

		while (fcd->rawframe != NULL)
		{
			verbose(" ...%d", fcd->rawframe->frames_behind);
			dc1394_capture_enqueue(fcd->camera, fcd->rawframe);
			dc1394_capture_dequeue(fcd->camera,
					DC1394_CAPTURE_POLICY_POLL,
					&(fcd->rawframe));
		}
		verbose("...done\n");
	}
	return DC1394_SUCCESS;
}

int FV_setFeatureRegister(FV_FirewireCamera* fcd, int featureId, uint value)
{
	verbose("FV_setFeatureRegister: %s\n", dc1394_feature_get_string(featureId));
	if (fcd)
	{
		if (featureId == DC1394_FEATURE_SHUTTER ||
			featureId == DC1394_FEATURE_GAIN)
		{
			dc1394_feature_set_mode(fcd->camera, DC1394_FEATURE_EXPOSURE, DC1394_FEATURE_MODE_MANUAL);
		}
		dc1394_feature_set_mode(fcd->camera, featureId, DC1394_FEATURE_MODE_MANUAL);
		dc1394_feature_set_absolute_control(fcd->camera, featureId, DC1394_OFF);
		dc1394_feature_set_value(fcd->camera, featureId, value);
	}
	return DC1394_SUCCESS;
}

int FV_getFeatureRegister(FV_FirewireCamera* fcd, int featureId, uint* value)
{
	verbose("FV_getFeatureRegister: %s\n", dc1394_feature_get_string(featureId));
	if (fcd)
	{
		dc1394_feature_get_value(fcd->camera, featureId, value);
	}
	return DC1394_SUCCESS;
}

int FV_setFeatureAbsolute(FV_FirewireCamera* fcd, int featureId, float value)
{
	verbose("FV_setFeatureAbsolute: %s\n", dc1394_feature_get_string(featureId));
	if (fcd)
	{
		if (featureId == DC1394_FEATURE_SHUTTER ||
			featureId == DC1394_FEATURE_GAIN)
		{
			dc1394_feature_set_mode(fcd->camera, DC1394_FEATURE_EXPOSURE, DC1394_FEATURE_MODE_MANUAL);
		}
		dc1394_feature_set_mode(fcd->camera, featureId, DC1394_FEATURE_MODE_MANUAL);
		dc1394_feature_set_absolute_control(fcd->camera, featureId, DC1394_ON);
		dc1394_feature_set_absolute_value(fcd->camera, featureId, value);
	}
	return DC1394_SUCCESS;
}

int FV_getFeatureAbsolute(FV_FirewireCamera* fcd, int featureId, float* value)
{
	verbose("FV_getFeatureAbsolute: %s\n", dc1394_feature_get_string(featureId));
	if (fcd)
	{
		dc1394_feature_get_absolute_value(fcd->camera, featureId, value);
	}
	return DC1394_SUCCESS;
}

int FV_setWhiteBalanceRegister(FV_FirewireCamera *fcd, uint blueValue, uint redValue)
{
	verbose("FV_setWhiteBalanceRegister\n");
	if (fcd)
	{
		dc1394_feature_set_mode(fcd->camera, DC1394_FEATURE_WHITE_BALANCE, DC1394_FEATURE_MODE_MANUAL);
		dc1394_feature_set_absolute_control(fcd->camera, DC1394_FEATURE_WHITE_BALANCE, DC1394_OFF);
		dc1394_feature_whitebalance_set_value(fcd->camera, blueValue, redValue);
	}
	return DC1394_SUCCESS;
}

int FV_getWhiteBalanceRegister(FV_FirewireCamera *fcd, uint* blueValue, uint* redValue)
{
	verbose("FV_getWhiteBalanceRegister\n");
	if (fcd)
	{
		dc1394_feature_whitebalance_get_value(fcd->camera, blueValue, redValue);
	}
	return DC1394_SUCCESS;
}

int FV_setFeaturePower(FV_FirewireCamera* fcd, int featureId, int power)
{
	verbose("FV_setFeaturePower: %s %4s\n", dc1394_feature_get_string(featureId), power ? "on" : "off");
	if (fcd)
	{
		dc1394_feature_set_power(fcd->camera, featureId, power ? DC1394_ON : DC1394_OFF);
	}
	return DC1394_SUCCESS;
}

int FV_getFeaturePower(FV_FirewireCamera* fcd, int featureId, int* power)
{
	verbose("FV_getFeaturePower: %s\n", dc1394_feature_get_string(featureId));
	if (fcd)
	{
		dc1394switch_t enumPower;
		dc1394_feature_get_power(fcd->camera, featureId, &enumPower);
		*power = (int) enumPower;
	}
	return DC1394_SUCCESS;
}

int FV_setFeatureMode(FV_FirewireCamera* fcd, int featureId, int mode)
{
	verbose("FV_setFeatureMode: %s\n", dc1394_feature_get_string(featureId));
	if (fcd)
	{
		dc1394_feature_set_mode(fcd->camera, featureId, mode);
	}
	return DC1394_SUCCESS;
}

int FV_getFeatureMode(FV_FirewireCamera* fcd, int featureId, int* mode)
{
	verbose("FV_getFeatureMode: %s\n", dc1394_feature_get_string(featureId));
	if (fcd)
	{
		dc1394feature_mode_t enumMode;
		dc1394_feature_get_mode(fcd->camera, featureId, &enumMode);
		*mode = (int) enumMode;
	}
	return DC1394_SUCCESS;
}

int FV_getColorFilter(FV_FirewireCamera* fcd, int* filter)
{
	verbose("FV_getColorFilter\n");
	if (fcd)
	{
		dc1394color_filter_t enumFilter = 0;
		dc1394_format7_get_color_filter(fcd->camera, fcd->mode, &enumFilter);
		*filter = (int) enumFilter;

	}
	return DC1394_SUCCESS;
}

/**
 * Enable or disable auto exposure.  Auto exposure is controlled by
 * exposure value (EV), shutter, and gain.  This function sets all those
 * features to auto mode if enable is non-zero, or to manual if enable
 * is zero.
 */
int FV_setAutoExposure(FV_FirewireCamera* fcd, int enable)
{
	verbose("FV_setAutoExposure: %6s\n", enable ? "true" : "false");
	if (fcd)
	{
		dc1394feature_mode_t fmode = enable ? DC1394_FEATURE_MODE_AUTO : DC1394_FEATURE_MODE_MANUAL;
		dc1394_feature_set_mode(fcd->camera, DC1394_FEATURE_EXPOSURE, fmode);
		dc1394_feature_set_mode(fcd->camera, DC1394_FEATURE_SHUTTER, fmode);
		dc1394_feature_set_mode(fcd->camera, DC1394_FEATURE_GAIN, fmode);
	}
	return DC1394_SUCCESS;
}

int FV_getAllFeatureInfo(FV_FirewireCamera* fcd, uint* registerValues, float* absoluteValues)
{
	verbose("FV_getAllFeatureInfo\n");
	if (fcd)
	{
		// get all features from camera
		dc1394featureset_t features;
		dc1394feature_info_t feature;
		dc1394_feature_get_all(fcd->camera, &features);

		// copy feature values into arrays
		int baseReg, baseAbs, i;
		for (i = 0; i < DC1394_FEATURE_NUM; i++)
		{
			feature = features.feature[i];
			baseReg = (feature.id - DC1394_FEATURE_MIN) * 5;
			baseAbs = (feature.id - DC1394_FEATURE_MIN) * 4;

			// set register values
			registerValues[baseReg + 3] = feature.available;
			registerValues[baseReg + 4] = feature.is_on;
			if (registerValues[baseReg + 3])
			{
				registerValues[baseReg + 0] = feature.value;
				registerValues[baseReg + 1] = feature.min;
				registerValues[baseReg + 2] = feature.max;
			}

			// set absolute values, if possible
			// need to store a boolean value in a float
			absoluteValues[baseAbs + 3] = feature.available && feature.absolute_capable ? 1.0 : 0.0;
			if (absoluteValues[baseAbs + 3])
			{
				absoluteValues[baseAbs + 0] = feature.abs_value;
				absoluteValues[baseAbs + 1] = feature.abs_min;
				absoluteValues[baseAbs + 2] = feature.abs_max;
			}

//			printf("%s  reg: %lu (%lu)  abs: %f (%f)\n", dc1394_feature_get_string(feature.id), registerValues[base + 0], registerValues[base + 3], absoluteValues[base + 0], absoluteValues[base + 3]);
		}
	}

	return DC1394_SUCCESS;
}

void FV_printFeatures(FV_FirewireCamera* fcd)
{
	if (fcd)
	{
	    verbose("FV_printFeatures: %p\n", fcd);
		dc1394featureset_t features;
		dc1394_feature_get_all(fcd->camera, &features);
		dc1394_feature_print_all(&features, stdout);
	}
}

/*
 *  Releases the camera, frees all allocated memory and exits
 */
void FV_closeVideoDevice(FV_FirewireCamera* fcd)
{
	verbose("FV_closeVideoDevice\n");
	if (fcd)
	{
		if (fcd->camera)
		{
			dc1394_video_set_transmission(fcd->camera, DC1394_OFF);
			dc1394_capture_stop(fcd->camera);
			dc1394_camera_free(fcd->camera);
		}

		if (fcd->fdevice)
			dc1394_free(fcd->fdevice);

		if (fcd->rgbframe && fcd->rgbframe->image)
			free(fcd->rgbframe->image);
		if (fcd->rgbframe)
			free(fcd->rgbframe);

		free(fcd);
	}
}

/**
 *  Sample rgb frame allocation function.
 */
void* allocateFrame(int* shape, int bytesPerPixel)
{
	int height = shape[0];
	int width = shape[1];
	int channels = shape[2];

	verbose("allocateFrame: %d x %d x %d of %d bytes\n", width, height, channels, bytesPerPixel);

	void* frame = malloc(bytesPerPixel * height * width * channels);
	return frame;
}

/**
 * Swap between big endian and little endian image data.
 */
void swapByteOrder(dc1394video_frame_t* frame)
{
//	verbose("swapByteOrder()\n");
	// this function only applies to RAW16, MONO16 and RGB16 data
	int channels =
			frame->color_coding == DC1394_COLOR_CODING_RGB16 ? 3 : 1;

	// get references to the image data memory
	uint16* usdata = (uint16*) frame->image;
	uint pixels = frame->size[0] * frame->size[1] * channels;

	// do a pair-wise swap of bytes in the image data
	int i;
	for (i = 0; i < pixels; i++)
	{
		usdata[i] = ((usdata[i] & 0x00ff) << 8) | ((usdata[i] & 0xff00) >> 8);
	}
}

void shiftBits(dc1394video_frame_t* frame, uint bits)
{
//	verbose("shiftBits()\n");
	// this function only applies to RAW16, MONO16 and RGB16 data
	int channels =
			frame->color_coding == DC1394_COLOR_CODING_RGB16 ? 3 : 1;

	// get references to the image data memory
	uint16* usdata = (uint16*) frame->image;
	int pixels = frame->size[0] * frame->size[1] * channels;

	// shift the bits in each pixel
	int i;
	for (i = 0; i < pixels; i++)
	{
		usdata[i] = usdata[i] >> bits;
	}
}

void printBinary(void* data, unsigned int size)
{
	// interpret data as an array of bytes
	unsigned char* byteData = (unsigned char*) data;

	// step along each byte in the data
	unsigned int i;
	for (i = 0; i < size; i++)
	{
		if (i % 8 == 0)
			printf("\n0x%04X: ", i);
		else
			putchar(' ');

		// print the bits in each byte using a mask
		unsigned char byte = byteData[i];
		unsigned char mask;
		for (mask = 0x80; mask != 0x00; mask >>= 1)
		{
			putchar(byte & mask ? '1' : '0');
		}
	}
	printf("\n");
}

/**
 * Print image info.
 */
void imageInfo(dc1394video_frame_t* frame)
{
	int width = frame->size[0];
	int height = frame->size[1];
	int channels =
			frame->color_coding == DC1394_COLOR_CODING_RGB8 ||
			frame->color_coding == DC1394_COLOR_CODING_RGB16 ? 3 :
			frame->color_coding == DC1394_COLOR_CODING_YUV422 ? 2 : 1;

	int pixels = width * height * channels;
	int is16bit =
			frame->color_coding == DC1394_COLOR_CODING_RAW16 ||
			frame->color_coding == DC1394_COLOR_CODING_MONO16;

	printf("image\n\tsize: %d x %d x %d\n", width, height, channels);
	printf("\tbytes in...\n");
	printf("\t    pixel:   %10d\n", is16bit ? 2 : 1);
	printf("\t    data:    %10u\n", frame->image_bytes);
	printf("\t    total:   %10llu\n", frame->total_bytes);
	printf("\t    padding: %10d\n", frame->padding_bytes);
	printf("\t    packet:  %10d\n", frame->packet_size);
	printf("\tpackets:     %10d\n", frame->packets_per_frame);
	printf("\tstride: %d\n", frame->stride);
	printf("\tfilter: %u\n", frame->color_filter);

	uint8*  b08data = (uint8*)  frame->image;
	uint16* b16data = (uint16*) frame->image;
//	uint32* b32data = (uint32*) frame->image;
//
//	printf("start 08: %02x %02x %02x %02x\n", b08data[0], b08data[1], b08data[2], b08data[3]);
//	printf("start 16: %04x %04x %04x %04x\n", b16data[0], b16data[1], b16data[2], b16data[3]);
//	printf("start 32: %08x %08x %08x %08x\n", b32data[0], b32data[1], b32data[2], b32data[3]);

	uint16 value = 0;
	uint32 min=1e6, max=0;
	uint32 sum=0, above=0;

	int i;
	uint8 first = 1;
	for (i = 0; i < pixels; i++)
	{
		value = is16bit ? b16data[i] : b08data[i];
		min = value < min ? value : min;
		max = value > max ? value : max;
		sum += value;

		above += value > 0x0FFF ? 1 : 0;
	}

	printf("\tmin: %u\n\tmax: %u\n\tmean: %5.3f\n", min, max, (float) sum / pixels);
	printf("\tabove 0x0FFF: %u\n", above);
}

float computeFramerate(FV_FirewireCamera* fcd, uint colorCoding, uint packetBytes)
{
	// determine data transfer parameters
//	dc1394speed_t isoSpeed;
//	dc1394_video_get_iso_speed(fcd->camera, &isoSpeed);
//	float packetTime = 500e-6 / pow(2, isoSpeed); // behavior does not match the description for 1394b
	float packetTime = 125e-6;
	uint bitsPerPixel;
	dc1394_get_color_coding_bit_size(colorCoding, &bitsPerPixel);

//	int packets = (fcd->width * fcd->height * pixelBits + 8 * packetBytes - 1) / (8 * packetBytes); // douxchamps
	int packets = ceil((float) fcd->width * fcd->height * bitsPerPixel / (8 * packetBytes));
	float framerate = 1.0 / (packetTime * packets);

	verbose("%d packets at %5.3f fps\n", packets, framerate);
	return framerate;
}

uint computePacketSize(FV_FirewireCamera* fcd, uint colorCoding, float framerate)
{
	// determine data transfer parameters
//	dc1394speed_t isoSpeed;
//	dc1394_video_get_iso_speed(fcd->camera, &isoSpeed);
//	float packetTime = 500e-6 / pow(2, isoSpeed); // behavior does not match the description for 1394b
	float packetTime = 125e-6;
	uint bitsPerPixel;
	dc1394_get_color_coding_bit_size(colorCoding, &bitsPerPixel);

	// determine range of possible packet sizes
	uint minPacket, maxPacket, recPacket;
	dc1394_format7_get_packet_parameters(fcd->camera, fcd->mode, &minPacket, &maxPacket);
	dc1394_format7_get_recommended_packet_size(fcd->camera, fcd->mode, &recPacket);
	if (framerate == 0)
		framerate = computeFramerate(fcd, colorCoding, maxPacket);

	// compute a recommended packet size
//	int packets = (int) (1.0 / (packetTime * framerate) + 0.5); // douxchamps
//	uint packetBytes = (fcd->width * fcd->height * bitsPerPixel + 8 * packets - 1) / (8 * packets);
	int packets = round(1.0 / (packetTime * framerate));
	float frameBytes = (float) fcd->width * fcd->height * bitsPerPixel / 8;
	uint packetBytes = ceil(frameBytes / packets);

	// validate the computed packet size
 	if (packetBytes > maxPacket)
 		packetBytes = maxPacket;
 	packetBytes -= packetBytes % minPacket;

 	packets = ceil(frameBytes / packetBytes);
 	int padding = (packetBytes * packets) - frameBytes;

 	// todo: try to minimize the padding required
	verbose("bytes per packet: [%d <= Bpp <= %d] rec'd %d, comp'd %d\n", minPacket, maxPacket, recPacket, packetBytes);
	verbose("%d packets, %d padding\n", packets, padding);
	return packetBytes;
}

