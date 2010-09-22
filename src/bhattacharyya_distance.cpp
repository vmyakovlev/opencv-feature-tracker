/* This is the contributed code:

Original Version: 2010-09-21  Michael Fang airfang613@gmail.com
Original Comments:

Implements Bhattacharyya distance

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

#include "distance.h"

// http://www.inf.ed.ac.uk/teaching/courses/av/MATLAB/COLOUR/bhattacharyya.m
float BhattacharyyaDistance::operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2 )
{
	CV_Assert(descriptor1.type() == CV_32F);
	CV_Assert(descriptor2.type() == CV_32F);
	int num_of_bins = descriptor1.cols;
	CV_Assert(num_of_bins == descriptor2.cols);

	float sum1 = 0.0f, sum2 = 0.0f;
	float bcoeff = 0.0f;
	for (int i = 0; i < num_of_bins; ++i) {
		bcoeff = bcoeff + sqrt(descriptor1.at<float>(0,i) * descriptor2.at<float>(0,i));
		sum1 += descriptor1.at<float>(0,i);
		sum2 += descriptor2.at<float>(0,i);
	}

	int sum_int = floor(sum1+0.5);
	CV_Assert(sum_int == floor(sum2+0.5));

	// a hack to overcome numerical error
	bcoeff = bcoeff > sum_int ? sum_int : bcoeff;

	float bdist = sqrt(sum_int - bcoeff);

	return bdist;
}

float BhattacharyyaDistance::operator()( const cv::MatND& descriptor1, const cv::MatND& descriptor2 )
{
	CV_Assert(descriptor1.type() == CV_32F);
	CV_Assert(descriptor2.type() == CV_32F);
	int dims = descriptor1.dims;
	CV_Assert(dims == 3);
	CV_Assert(dims == descriptor2.dims);
	int dim1 = descriptor1.size[0];
	int dim2 = descriptor1.size[1];
	int dim3 = descriptor1.size[2];
	int num_of_bins = dim1 * dim2 * dim3;

	float bcoeff = 0.0f;
	for (int i = 0; i < dim1; ++i) 
		for (int j = 0; j < dim2; ++j)
			for (int k = 0; k < dim3 ; ++k) {
				float descr1_elem = descriptor1.at<float>(i,j,k);
				float descr2_elem = descriptor2.at<float>(i,j,k);
				bcoeff += sqrtf(descr1_elem * descr2_elem);
			}

	float sum1 = sum(descriptor1)[0];
	float sum2 = sum(descriptor2)[0];
	CV_Assert(sum1 == sum2);

	// a hack to overcome numerical error
	bcoeff = bcoeff > sum1 ? sum1 : bcoeff;

	float bdist = sqrt(sum1 - bcoeff);

	return bdist;
}
