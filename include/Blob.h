#ifndef __BLOB_H
#define __BLOB_H
#include <cv.h>
#include <vector>
#include <string>
using namespace cv;

/** \class Blob
  A blob can be defined by its contour. However, having a separate blob class is a mental
  indication that we are dealing with this contour as an object. That is, we are interested in this
  object's area, its center of mass, its enclosing rectangle ... .
*/
class Blob{
public:
    Blob();
    Blob(const std::vector<Point2f> & contour_points);
    ~Blob();
    double Area() const;
    RotatedRect GetBoundingRectangle() const;
    Rect GetBoundingUprightRectangle() const;

    // Conversion to other objects
    operator KeyPoint() const;

    // Visualization helper methods
    void DrawTo(Mat im, const std::string custom_msg = "", const cv::Scalar & color = CV_RGB(0,255,0)) const;
private:
    std::vector<Point2f> points_; //!< points that make up this contour
    RotatedRect bounding_rotated_rect_; //!< a minimum-area bounding rotated rectangle
    double area_; //!< blob area
};

#endif
