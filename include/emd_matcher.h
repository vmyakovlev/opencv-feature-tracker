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
#ifndef __EMD_MATCHER_H
#define __EMD_MATCHER_H

#include <cv.h>
#include "feature.h"
#include <exception>

/** \class EMDDescriptorMatcher
    \todo Make distance option settable
    \todo Make diagonal image length settable
  */
template<class DistanceType>
class EMDDescriptorMatcher : public DescriptorMatcher
{
public:
    EMDDescriptorMatcher(DistanceType distance = DistanceType() );
    virtual void index(const std::vector<cv::KeyPoint>& db_keypoints,
                       const cv::Mat& db_descriptors);
    virtual void match(const std::vector<cv::KeyPoint>& query_keypoints,
                       const cv::Mat& query_descriptors,
                       std::vector<int>& matches,
                       std::vector<float>& distance) const;
    void SetWeights(const cv::Mat & weight1, const cv::Mat & weight2);
private:
    // These signature matrices are to be set up so that
    // cvCalcEMD2 can be used
    cv::Mat signature1_; //!< Signature matrix for descriptor 1
    cv::Mat signature2_; //!< Signature matrix for descriptor 2

    cv::Mat weight1_;
    cv::Mat weight2_;

    // Save these information from index() so they can be used in match()
    std::vector<cv::KeyPoint> db_keypoints_;
    cv::Mat db_descriptors_;
};

#endif
