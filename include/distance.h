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
#ifndef __DISTANCE_H_
#define __DISTANCE_H_

#include <cv.h>

/** \class Distance

  Typically, one uses an implementation of this interface in order to compute
  distance between two descriptors.

  The standard descriptor input are 1xN in size. That is, the descriptor is a
  horizontal vector. This is because it fits into how we keep our descriptors
  in memory after getting them from a DescriptorExtractor.

  The operator() method can be made static in this interface but that will not
  allow implementations to accessing member variables. A parametric distance
  method might requires such capability.

  Your implementation should always support the standard usage:

    DescriptorExtractor descriptor_extractor;
    Mat descriptors1, descriptors2;

    descriptor_extractor.compute(input_image, descriptors1);
    descriptor_extractor.compute(input_image2, descriptors2);

    Distance distance;
    for (int i=0; i<descriptors1.rows; ++i){
        std::cout << "Distance " << distance(descriptors1.row(i), descriptors2.row(i))
                  << std::endl;
    }

  As you can see, the input descriptor is 1xN in size. However, we leave the size
  enforcement to your implementation.

  In some cases, individual distance between two descriptors are combined into
  another distance. This is called distance fusion and is provided by a DistanceFuser.

  */
class Distance
{
    virtual float operator()(const cv::Mat& descriptor1, const cv::Mat& descriptor2) = 0;
};

class L1Distance : public Distance
{
    float operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2){
        return cv::norm(descriptor1 - descriptor2, cv::NORM_L1);
    }
};

class L2Distance : public Distance
{
public:
    float operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2){
        return cv::norm(descriptor1 - descriptor2, cv::NORM_L2);
    }
};

class BhattacharyyaDistance : public Distance
{
public:
    float operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2);
};

class PearsonCoefficientDistance: public Distance
{
public:
    float operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2);
};

#endif
