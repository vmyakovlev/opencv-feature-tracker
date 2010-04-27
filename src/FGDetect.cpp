#include "FGDetect.h"
FGDetect::FGDetect(){
	IplConvKernel* struc= cvCreateStructuringElementEx(3, 13, 1, 6, CV_SHAPE_RECT, NULL) ;
	IplConvKernel* struc2 = cvCreateStructuringElementEx(3, 13, 1, 6, CV_SHAPE_RECT, NULL) ;
}
void FGDetect::fgdetec(IplImage *img, vector<CvRect> bbox)
{
	cvSmooth(img,img,CV_GAUSSIAN,5,5,1);
	Mat a1=Mat(img);
	Mat a2=Mat(src);

	Mat diff_mat = abs(a1-a2);
	vector<Mat> planes;
	split(diff_mat,planes);
	Mat sum_mat = planes[0] + planes[1] + planes[2];
	Mat output = sum_mat > 70;
	IplImage img2=output;
	IplImage img3=img2;
	cvMorphologyEx(&img2,&img2,&img3,struc,CV_MOP_OPEN);
	cvMorphologyEx(&img2,&img2,&img3,struc,CV_MOP_CLOSE);
	cvDilate(&img2,&img2,struc2);
	CvSeq* contour = 0;
	cvThreshold( &img2, &img2, 1, 255, CV_THRESH_BINARY );
	CBlobResult blobs;
	int i;
	CBlob *currentBlob;
	blobs = CBlobResult( &img2, NULL, 256);
	blobs.Filter( blobs, B_EXCLUDE, CBlobGetArea(), B_LESS, 500 );
	for (i = 0; i < blobs.GetNumBlobs(); i++ ){
		currentBlob = blobs.GetBlob(i);
		CvRect rec=currentBlob->GetBoundingBox();
		CvPoint p1,p2;
		p1.x=rec.x;
		p1.y=rec.y-5;
		p2.x=rec.x+rec.width;
		p2.y=rec.y+rec.height;
		cvDrawRect(img,p1,p2,CV_RGB(255,0,0));
		bbox.push_back(rec);
	}
}

