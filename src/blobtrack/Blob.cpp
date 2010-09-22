/* This is the contributed code:

Original Version: 2010-09-21  Dat Chu dattanchu@gmail.com
Original Comments:

Implement a simple Blob class for describing an image blob

*/

/*///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//*/

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
void Blob::DrawTo(Mat im, const std::string custom_message /* = "" */, const cv::Scalar & color/* = CV_RGB(0,255,0) */) const{
    draw_hull(im, points_, CV_RGB(255,0,0));

    // blob minAreaRect drawn as an Ellipse
    rotated_rect(im, bounding_rotated_rect_, color);

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

/** \brief Similar to get bounding box but returns an upright rectangle instead of a rotated one
*/
Rect Blob::GetBoundingUprightRectangle() const{
    return GetBoundingRectangle().boundingRect();
}

RotatedRect Blob::GetBoundingRectangle() const{
    return bounding_rotated_rect_;
}
