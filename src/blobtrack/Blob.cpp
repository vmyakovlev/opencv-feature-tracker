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

/** \brief Create a blob object given its contour points
*/
Blob::Blob(const vector<Point2f> & contour_points ){
    points_ = contour_points;

    Mat points(contour_points);

    // find bounding rectangle
    vector<Point2f> hull;
    convexHull(points, hull);
    Mat hull_points(hull);
    bounding_rotated_rect_ =  minAreaRect(hull_points); // internally minAreaRect compute convex hull so oh well :D

    // find the areas
    area_ = contourArea(points);
}

Blob::~Blob(){}

/** \brief Draw this blob to the input image
*/
void Blob::DrawTo(Mat im, const std::string custom_message) const{
    draw_hull(im, points_, CV_RGB(255,0,0));

    // blob minAreaRect drawn as an Ellipse
    rotated_rect(im, bounding_rotated_rect_, CV_RGB(0,255,0));

    // blob center
    circle(im, bounding_rotated_rect_.center, 1, CV_RGB(0,0,255));

    // Write a custom text
    if (!custom_message.empty()){
        cv::putText(im, custom_message, bounding_rotated_rect_.center, FONT_HERSHEY_PLAIN, 1, CV_RGB(255,255,255));
    }
}

/** \brief Get you the area of this blob
*/
double Blob::Area() const{
    return area_;
}

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
