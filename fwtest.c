#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <firewireVideo.h>

/**
 * A simple test of the firewireVideo library.  Opens a firewire camera
 * and saves some number of frames to disk.  Usage:
 *
 * fwtest count mode
 *
 * count - the number of frames to save to disk
 * mode  - the libdc1394 video mode code to use for the camera this
 * 		defaults to DC1394_VIDEO_MODE_640x480_RGB8 (68).
 */
int main(int argc, char *argv[]) 
{
	FV_FirewireCamera *cam;
	void* frame;
	FILE *fp;
	char filename[256];

	uint8 var8;
	uint16 var16;
	uint32 var32;
	printf("sizeof uint8:  %ld\n", sizeof(var8));
	printf("sizeof uint16: %ld\n", sizeof(var16));
	printf("sizeof uint32: %ld\n", sizeof(var32));
	printf("sizeof int:    %ld\n", sizeof(int));
    printf("sizeof long:   %ld\n", sizeof(long));
    printf("sizeof long uint: %ld\n", sizeof(long unsigned int));
    printf("sizeof int*:   %ld\n", sizeof(int*));

	// parse command line arguments
	int frames = 1;
	dc1394video_mode_t mode = DC1394_VIDEO_MODE_640x480_RGB8;
	if (argc > 1)
		frames = atoi(argv[1]);
	if (argc > 2)
		mode = atoi(argv[2]);

	printf("Opening device\n");
	cam = FV_openVideoDevice(0, DC1394_ISO_SPEED_400, 10);
	if (!cam) return -1;

	FV_setVideoMode(cam, mode, 0);
//	FV_setVideoMode(cam, DC1394_VIDEO_MODE_800x600_RGB8, DC1394_FRAMERATE_15);
//	FV_setVideoModeF7(cam, DC1394_VIDEO_MODE_FORMAT7_0, DC1394_COLOR_CODING_RAW16, 400, 400, 0, 0, 0, SWAP_BYTES);
	FV_setAutoExposure(cam, 0);
//	FV_printFeatures(cam);
	FV_startTransmission(cam);

	FV_printFeatures(cam);

	printf("Capturing %d images\n", frames);
	int i=0;
	time_t t0, t1;
	time(&t0);

	for (i = 0; i < frames; i++)
	{
		// grab frame
		frame = FV_acquireFrame(cam, allocateFrame);

		// save frame
		sprintf(filename, "image-%03d.ppm", i);
		fp = fopen(filename, "w");
		fprintf(fp, "P6\n%d %d\n%d\n", cam->width, cam->height, 255);
		fwrite(frame, sizeof(unsigned char), 3 * cam->width * cam->height, fp);
//		fprintf(fp, "P5\n%d %d\n%d\n", cam->width, cam->height, 255);
//		fwrite( frame, sizeof(unsigned char), 1 * cam->width * cam->height, fp );
//		fprintf(fp, "P5\n%d %d\n%d\n", cam->width, cam->height, 4095);
//		fwrite( frame, sizeof(unsigned short int), 1 * cam->width * cam->height, fp );

		fclose(fp);
		free(frame);

		// report frame rate
		if ((i+1) % 10 == 0)
		{
			time(&t1);
			printf("%3d:\t%8.3f fps\t%d lag\n", (i+1), ((double) (i+1) / (t1-t0+1e-6)), cam->rawframe->frames_behind);
		}
	}

	printf("Closing video device\n");
	FV_closeVideoDevice(cam);
	return(0);
}
