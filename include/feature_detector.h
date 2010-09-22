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
#ifndef __FEATURE_DETECTOR_H_
#define __FEATURE_DETECTOR_H_

#include "cv.h"
#include "feature.h"

class ShiTomashiFeatureDetector : public ::FeatureDetector{
public:
    ShiTomashiFeatureDetector(int max_corners = 100, double quality_level = 0.1, double min_distance = 5,
                              int block_size = 3, bool use_harris_detector = false, double k = 0.04) :
            max_corners_(max_corners), quality_level_(quality_level), min_distance_(min_distance),
            block_size_(block_size), use_harris_detector_(use_harris_detector), k_(k){
    }

    virtual void detect(const cv::Mat& image,
                        std::vector<cv::KeyPoint>& keypoints,
                        const cv::Mat& mask = cv::Mat() ){

        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(image, corners, max_corners_, quality_level_, min_distance_, mask, block_size_, use_harris_detector_, k_ );

        // convert these Point2f into cv::KeyPoint
        keypoints.clear();
        keypoints.resize(corners.size());

        for (int i=0; i<keypoints.size(); ++i){
            keypoints[i] = cv::KeyPoint(corners[i], 1);
        }
    }

    int max_corners_;
    double quality_level_;
    double min_distance_;
    int block_size_;
    bool use_harris_detector_;
    double k_;
};

#endif
