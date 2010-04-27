#include "Blob.h"
#include "draw.h"

using std::vector;
using namespace cv;

/** \brief Empty constructor

  So that you can use me easier within container
*/
Blob::Blob(){
    area_ = 0;
}

/** \brief Create a blob object given input detected points from MSER
*/
Blob::Blob(const vector<Point> & mser_points ){
    Mat points(mser_points);
    convexHull(points, hull_);
    Mat hull_points(hull_);
    bounding_rotated_rect_ =  minAreaRect(hull_points); // internally minAreaRect compute convex hull so oh well :D
    area_ = contourArea(points);
}

Blob::~Blob(){}

/** \brief Draw this blob to the input image
*/
void Blob::draw_to(Mat im) const{
    draw_hull(im,hull_, CV_RGB(255,0,0));

    // blob minAreaRect drawn as an Ellipse
    rotated_rect(im, bounding_rotated_rect_, CV_RGB(0,255,0));

    // blob center
    circle(im, bounding_rotated_rect_.center, 1, CV_RGB(0,0,255));
}

/** \brief Get you the area of this blob
*/
double Blob::area() { return area_;}

/** \brief Conversion to keypoint to fit into keypoint

  This way our blob class can fit into the framework of feature detection/extraction
*/
Blob::operator KeyPoint() const{
    return KeyPoint(bounding_rotated_rect_.center,
                    max(bounding_rotated_rect_.size.height, bounding_rotated_rect_.size.width),
                    bounding_rotated_rect_.angle, // angle is that of rotated rectangle angle
                    bounding_rotated_rect_.size.area() // response strength is similar to area
                    );
}
