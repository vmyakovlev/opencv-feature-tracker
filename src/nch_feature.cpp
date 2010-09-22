/* This is the contributed code:

Original Version: 2010-09-21  Michael Fang
Original Comments:

Normalized Color Histogram feature descriptor extractor

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
#include "nch_feature.h"
#include <iostream>
using namespace cv;
using std::cout;
using std::endl;

NCHDescriptorExtractor::NCHDescriptorExtractor()
{
	bins_ = 8;
}

NCHDescriptorExtractor::NCHDescriptorExtractor( int bins )
	:bins_(bins)
{}

NCHDescriptorExtractor::~NCHDescriptorExtractor() {}

void NCHDescriptorExtractor::compute( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors ) const
{
	cout << "FATAL ERROR: Class NCHFeatureExtractor does not have method compute() implemented." << endl
		 << "Method compute_dense() should be called instead." << endl;
    CV_Error(CV_StsNotImplemented, "This descriptor does not support compute()");
}

void NCHDescriptorExtractor::compute_dense( const cv::Mat& image, cv::MatND& descriptor ) const
{
	// assume the input image is 3-channel
	CV_Assert(image.channels() == 3);
	Mat img;
	if (image.type() != CV_32FC3) {
		image.convertTo(img, CV_32FC3);
	}
	else img = image;

	// prepare the MatND histogram
	const int hist_size[] = {bins_, bins_, bins_};
	
	descriptor.create(3, hist_size, CV_32F);
	descriptor = Scalar(0);

	// accumulate the histogram
	int num_of_valid_pixels = 0;
	int bin_max = bins_ - 1;
	MatConstIterator_<Vec3f> it = img.begin<Vec3f>(),
							 it_end = img.end<Vec3f>();
	for ( ; it != it_end; ++it) {
		
		Vec3f pix;
		pix[0] = (*it)[0];
		pix[1] = (*it)[1];
		pix[2] = (*it)[2];
				
		if (pix[0] == -1 || pix[1] == -1 || pix[2] == -1) continue;
		else num_of_valid_pixels++;
		
		// normalize by removing brightness
		float sum = pix[0] + pix[1] + pix[2];
		if (sum != 0) {
			pix[0] = pix[0] / sum;
			pix[1] = pix[1] / sum;
			pix[2] = pix[2] / sum;
		}
		
		// compute index (range: 0 - bins-1)
		int idx1 = floor(pix[0]*bin_max+0.5);
		int idx2 = floor(pix[1]*bin_max+0.5);
		int idx3 = floor(pix[2]*bin_max+0.5);

		descriptor.at<float>(idx1,idx2,idx3) += 1.0f;
	}
	
	normalize(descriptor, descriptor, 1, 0, NORM_L1);
	
	CV_Assert(fabs(sum(descriptor)[0] - 1) < FLT_EPSILON);

    /* NOTE: Check Revision 339 for some code that was under #if 0
       @MF: if the code is useful but not current used, please put its purpose here
    */

}

void NCHDescriptorExtractor::compute_dense( const std::vector<cv::Mat>& images, std::vector<cv::MatND>& descriptors ) const
{
	int num_of_inputs = images.size();
	for (int i = 0; i < num_of_inputs; ++i) {
		MatND descriptor;
		compute_dense(images[i], descriptor);
		descriptors.push_back(descriptor);
	}
}

int NCHDescriptorExtractor::get_bins()
{
	return bins_;
}
