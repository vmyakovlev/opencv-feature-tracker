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
#ifndef WINDOW_PAIR_H_
#define WINDOW_PAIR_H_

#include <cv.h>
#include <string>

/** \class WindowPair
    A window that manage two image put side by side. Has methods that allow annotations
    of points on each image. Good for showing correspondence between two image.
*/
class WindowPair{
public:
    WindowPair(const cv::Mat & im1, const cv::Mat & im2, const std::string & name);
    void DrawArrow(cv::Point im1_from, cv::Point  im2_to, const cv::Scalar & color, int thickness=1, int lineType=8, int shift=0);
    int Show(int delay=0);
    void Save(const std::string & filename);
    cv::Mat get_image();
private:
    cv::Mat im_;
    cv::Point im2_offset; //!< the offset that takes a point in im2 coordinate to window coordinate
    std::string window_name_;
    float alpha_blending_; //!< alpha blending coefficient
};

#endif
