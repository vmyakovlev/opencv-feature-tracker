#ifndef __BLOB_H
#define __BLOB_H
#include <cv.h>
#include <vector>
using std::vector;
using namespace cv;

/** \class Blob
  A blob can be defined by its contour. However, having a separate blob class is a mental
  indication that we are dealing with this contour as an object. We are interested in this
  object's area, its center of mass, its enclosing rectangle ... .
*/
class Blob{
public:
    Blob();
    Blob(const vector<Point> & contour_points);
    ~Blob();
    double Area() const;
    RotatedRect GetBoundingRectangle() const;

    // Conversion to other objects
    operator KeyPoint() const;

    // Visualization helper methods
    void DrawTo(Mat im) const;
private:
    vector<Point> points_; //!< points that make up this contour
    RotatedRect bounding_rotated_rect_; //!< a minimum-area bounding rotated rectangle
    double area_; //!< blob area
};

#endif
