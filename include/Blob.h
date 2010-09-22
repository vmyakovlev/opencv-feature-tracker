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
