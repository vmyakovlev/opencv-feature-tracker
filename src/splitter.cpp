/* This is the contributed code:

Original Version: 2010-09-21  Michael Fang airfang613@gmail.com
Original Comments:

Implements a simple top-bottom splitter for a matrix

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
#include "splitter.h"
#include <iostream>

using cv::Mat;
using std::cout;
using std::endl;

void Splitter::SplitMat(const cv::Mat& mat_in, SplitMode mode, std::vector<cv::Mat>* mats_out) {
	Mat patch;
	int rows = mat_in.rows;
	int split_idx;
	switch (mode) {
		case NONE:
			//cout << "NONE: Not performing splitting." << endl;
			mats_out->push_back(mat_in);
			break;
		case UL:
			//cout << "UL: Performing top/bottom split." << endl;
			split_idx = floor(rows / 2.0f + 0.5f);
			patch = mat_in(cv::Range(0,split_idx), cv::Range::all());
			mats_out->push_back(patch);
			patch = mat_in(cv::Range(split_idx,rows), cv::Range::all());
			mats_out->push_back(patch);
			break;
		case ULF:
			cout << "ULF: Performing top/bottom/foot split." << endl;
			cout << "But too bad, this has not been implemented yet." << endl;
			break;
		default:
			cout << "UNKNOWN: Invalid split mode." << endl;
			break;
	}
};