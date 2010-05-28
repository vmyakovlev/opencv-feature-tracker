#ifndef __BLOB_H
#define __BLOB_H
#include <cv.h>
#include <vector>
using std::vector;
using namespace cv;

class Blob{
public:
    Blob();
    Blob(const vector<Point> & mser_points );
    ~Blob();
    void draw_to(Mat im) const;
    double area();
    operator KeyPoint() const;
private:
    vector<Point> hull_; //!< hull of the blob
    RotatedRect bounding_rotated_rect_; //!< a minimum-area bounding rotated rectangle
    double area_; //!< blob area
};

#endif
