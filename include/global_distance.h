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
#ifndef __GLOBAL_DISTANCE_H
#define __GLOBAL_DISTANCE_H

#include <cv.h>
#include <iostream>
#include <vector>

template <class DistT>
class GlobalDistance
{
public:
	float operator()(const std::vector<cv::Mat>& descriptors1, 
		const std::vector<cv::Mat>& descriptors2)
	{
		int num_of_inputs = descriptors1.size();
		CV_Assert(num_of_inputs == descriptors2.size());

		int channels = descriptors1[0].channels();
		CV_Assert(channels == descriptors2[0].channels());

		int dim = descriptors1[0].cols;
		CV_Assert(dim == descriptors2[0].cols);

		// MF: assuming the descriptor dimension is 1 row by N columns
		Mat descriptor1_global(1, num_of_inputs*dim, descriptors1[0].type());
		Mat descriptor2_global(1, num_of_inputs*dim, descriptors1[0].type());

		for (int i = 0; i < num_of_inputs; ++i) {
            Mat descriptors1_rows = descriptor1_global(Range::all(),Range(i*dim, (i+1)*dim));
            Mat descriptors2_rows = descriptor1_global(Range::all(),Range(i*dim, (i+1)*dim));

            descriptors1[i].copyTo(descriptors1_rows);
            descriptors2[i].copyTo(descriptors2_rows);
		}
		
		Mat *descriptor1 = new Mat[channels];
		Mat *descriptor2 = new Mat[channels];
		
		float res = 0.0f;
		for (int i = 0; i < channels; ++i) {
			split(descriptor1_global, descriptor1);
			split(descriptor2_global, descriptor2);
			
			float dist = distance_type(descriptor1[i], descriptor2[i]);
			// MF: naive way of fusing channel scores, i.e., averaging
			res += dist;
			//if (dist > res)
			//	res = dist;
		}
		res /= channels;
		
		delete [] descriptor1;
		delete [] descriptor2;

		return res;
	}

	float operator()(const std::vector<cv::MatND>& descriptors1, 
		const std::vector<cv::MatND>& descriptors2) {

		int num_of_inputs = descriptors1.size();
		CV_Assert(num_of_inputs == descriptors2.size());
		
		int dims = descriptors1[0].dims;
		CV_Assert(dims == 3);
		CV_Assert(dims == descriptors2[0].dims);
		
		int dim1 = descriptors1[0].size[0];
		int dim2 = descriptors1[0].size[1];
		int dim3 = descriptors1[0].size[2];
		const int new_size[] = {dim1, dim2, num_of_inputs * dim3};
		cv::MatND descriptor1_global(dims, new_size, descriptors1[0].type());
		cv::MatND descriptor2_global(dims, new_size, descriptors2[0].type());
		
		Range* ranges = new Range [dims];
		ranges[0] = Range::all();
		ranges[1] = Range::all();
		for (int i = 0; i < num_of_inputs; ++i) {
			ranges[2] = Range(i*dim3, (i+1)*dim3);
			MatND descriptors1_roi = descriptor1_global(ranges);
			MatND descriptors2_roi = descriptor2_global(ranges);

			descriptors1[i].copyTo(descriptors1_roi);
			descriptors2[i].copyTo(descriptors2_roi);
		}
		
		float res = distance_type(descriptor1_global, descriptor2_global);

		delete [] ranges;
		return res;
	}

private:
	DistT distance_type;
};

#endif
