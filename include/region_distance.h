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
#ifndef __REGION_SIMILARITY_H
#define __REGION_SIMILARITY_H

#include <cv.h>
#include <iostream>
#include <stdio.h>
#include <vector>

#include "misc.h"
using namespace cv;

template <class DistT>
class RegionDistance
{
public:
	float operator()(const std::vector<cv::Mat>& descriptors1, 
					 const std::vector<cv::Mat>& descriptors2)
	{
		int num_of_inputs = descriptors1.size();
		CV_Assert(num_of_inputs == descriptors2.size());

		int channels = descriptors1[0].channels();
		CV_Assert(channels == descriptors2[0].channels());

		Mat *descriptor1 = new Mat[channels];
		Mat *descriptor2 = new Mat[channels];
		Mat_<float> dist_sum(1, channels, 0.0f);

		for (int i = 0; i < num_of_inputs; ++i) {
			split(descriptors1[i], descriptor1);
			split(descriptors2[i], descriptor2);

			// MF: naive way of fusing part scores, i.e., averaging
			for (int j = 0; j < channels; ++j) {
				dist_sum[0][j] += distance_type(descriptor1[j], descriptor2[j]);
			}
		}		
		dist_sum /= num_of_inputs;
	
		float res = 0.0f;
		for (int i = 0; i < channels; ++i) {
			// MF: naive way of fusing channel scores, i.e., averaging
			res += dist_sum[0][i];
			//if (dist_sum[0][i] > res)
			//	res = dist_sum[0][i];
		}
		res /= channels;

		printf("R: %.3f G: %.3f B: %.3f\n",dist_sum[0][0],dist_sum[0][1],dist_sum[0][2]);

		delete [] descriptor1;
		delete [] descriptor2;

		return res;
	};

	float operator()(const std::vector<cv::MatND>& descriptors1, 
		const std::vector<cv::MatND>& descriptors2)
	{
		int num_of_inputs = descriptors1.size();
		CV_Assert(num_of_inputs == descriptors2.size());

		float res = 0.0f;

		for (int i = 0; i < num_of_inputs; ++i) {
			// MF: naive way of fusing part scores, i.e., averaging
			float dist = distance_type(descriptors1[i], descriptors2[i]);
			printf("Part %d : %.3f ", i+1, dist);
			res += dist;
		}		
		res /= num_of_inputs;
		printf("\n");

		return res;
	};

private:
	DistT distance_type;
};

#endif
