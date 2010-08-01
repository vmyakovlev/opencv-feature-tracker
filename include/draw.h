#ifndef __DRAW_H
#define __DRAW_H
#include <cv.h>
#include <vector>
using std::vector;
using namespace cv;

void draw_hull(Mat im, const vector<Point> hull_points, CvScalar color);
void draw_hull(Mat im, const vector<Point2f> hull_points, CvScalar color);
void rotated_rect(Mat im, const RotatedRect & rot_rect, CvScalar color);
void polylines(Mat im, vector<vector<Point2f> > points, bool isClosed, const Scalar & color, int thickness = 1, int lineType=8, int shift=0);

#endif
