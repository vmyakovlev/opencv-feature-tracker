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
#ifndef __DESCRIPTOR_MATCH_H
#define __DESCRIPTOR_MATCH_H

#include "feature.h"
#include <vector>
#include <cv.h>

class KLTTracker : public DescriptorMatchGeneric {
public:
    //! Adds keypoints from a single image to the training set (descriptors are supposed to be calculated here)
    virtual void add(const cv::Mat& image, std::vector<cv::KeyPoint>& points);

    //! Classifies test keypoints
    /**
      Does nothing in KLT tracker
      */
    virtual void classify(const cv::Mat& image, std::vector<cv::KeyPoint>& points){
        CV_Error(CV_StsNotImplemented, "This matcher doesn't support classifying. Use searching instead.");
    }

    //! Matches test keypoints to the training set
    virtual void match(const cv::Mat& image, std::vector<cv::KeyPoint>& points, std::vector<int>& indices){
        CV_Error(CV_StsNotImplemented, "This matcher doesn't support matching. Use searching instead.");
    }

    /**
        \see DescriptorMatchGeneric
    */
    virtual void search(const cv::Mat& test_image, std::vector<cv::Point2f>& output_found_points, std::vector<int>& training_point_indices);
private:
    cv::Mat source_image_;
    std::vector<cv::Point2f> source_points_;
};

#endif
