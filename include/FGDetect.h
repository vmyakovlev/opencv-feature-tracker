#ifndef FGDETECT_H
#define FGDETECT_h

#include <cv.h>
#include <highgui.h>
#include "BlobResult.h"
using namespace cv;
class FGDetect{
private:
	IplConvKernel *struc,*struc2;
	IplImage *src;
public:
	FGDetect();
	void setBG(IplImage *bg){src=bg;}
	void fgdetec(IplImage *img, vector<CvRect> bbox);
};
#endif