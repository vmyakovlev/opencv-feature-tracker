/* This is the contributed code:

Original Version: 2010-09-21  Dat Chu dattanchu@gmail.com
Original Comments:

Implements a window pair class for drawing two images side by side and draw arrows connecting
two points on those images.

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
#include "window_pair.h"
#include <cv.h>
#include <highgui.h>

using namespace cv;

WindowPair::WindowPair(const cv::Mat & im1, const cv::Mat & im2, const std::string & name)
{
    im_ = Mat::zeros(max(im1.rows,im2.rows), im1.cols + im2.cols, CV_8UC3);
    im2_offset = cv::Point(im1.cols,0);
    window_name_ = name;
    alpha_blending_ = 0.9;

    // copy image data to the right place
    cv::Mat im_im1 = cv::Mat(im_, cv::Range(0,im1.rows), cv::Range(0,im1.cols));
    cv::Mat im_im2 = cv::Mat(im_, cv::Range(0,im2.rows), cv::Range(im1.cols,im1.cols+im2.cols));

    // convert grayscale image as necessary
    cv::Mat im1_color, im2_color;
    if (im1.channels() == 1){
        cv::cvtColor(im1,im1_color, CV_GRAY2RGB);
    } else if (im1.channels() == 3) {
        im1_color = im1;
    } else {
        CV_Error(CV_StsBadSize, "input image 1 needs to be either single-channel or 3-channel");
    }

    if (im2.channels() == 1){
        cvtColor(im2,im2_color, CV_GRAY2RGB);
    } else if (im2.channels() == 3) {
        im2_color = im2;
    } else {
        CV_Error(CV_StsBadSize, "input image 2 needs to be either single-channel or 3-channel");
    }

    // Copy data to our internal image
    im1_color.copyTo(im_im1);
    im2_color.copyTo(im_im2);
}

/** \brief Draw an arrow from a point in im1 to a point in im2
  \param[in] im1_from the start point of the arrow in im1 coordinate
  \param[in] im2_to the end point of the arrow in im2 coordinate
*/
void WindowPair::DrawArrow(cv::Point im1_from, cv::Point im2_to, const cv::Scalar & color, int thickness, int lineType, int shift){
    cv::Mat new_layer = cv::Mat::zeros(im_.size(), CV_8UC3);
    cv::line(new_layer, im1_from, im2_offset + im2_to, color, thickness, lineType, shift);
    cv::addWeighted(new_layer, alpha_blending_, im_, 1,0.0,im_);
}

int WindowPair::Show(int delay){
    cv::namedWindow(window_name_,CV_WINDOW_AUTOSIZE);
    cv::imshow(window_name_,im_);
    return cv::waitKey(delay);
}

/** \brief Save output window pair image to filename
*/
void WindowPair::Save(const std::string & filename){
    cv::imwrite(filename, im_);
}

/** \brief Get access to the internal image
*/
cv::Mat WindowPair::get_image(){
    return im_;
}
