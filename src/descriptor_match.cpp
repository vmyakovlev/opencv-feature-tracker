/* This is the contributed code:

Original Version: 2010-09-21  Dat Chu dattanchu@gmail.com
Original Comments:

Implements KLT tracker with only the searching feature
NOTE: there seems to be a similar version implemented in OpenCV core. Should probably merge with it

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
#include "descriptor_match.h"
#include "misc.h"
using namespace cv;

//! Adds keypoints from this source image
/** For KLT Tracker, there is no need to extract any features, so we just save these input for later processing
  in match()
*/
void KLTTracker::add(const Mat& image, vector<KeyPoint>& points){
    source_image_ = image;
    vector_one_to_another(points, source_points_);
}

//! Matches test keypoints to the training set
/**
  This implementation group together feature extraction and matching by using OpenCV calcOpticalFlowPyrLK.
  In order to make this implementation more adapted to the new framework, one needs to dissect calcOpticalFlowPyrLK
  into the feature calculation component and the matching component.

  \param test_image input test image
  \param[out] output_found_points where we found the training points in this image
  \param[out] training_point_indices the corresponding indices of the training points for these found points
              e.g. [3,4,5] means the 1st, 2nd, and 3rd elements in output_found_points correspond to the 4th, 5th and 6th elements in
                           the training points. See add().

*/
void KLTTracker::search(const Mat& test_image, vector<Point2f>& output_found_points, vector<int>& training_point_indices){
    output_found_points.clear();
    training_point_indices.clear();
    vector<Point2f> target_points;
    vector<uchar> status;
    vector<float> err;

    // TODO: add warning messages when this is encountered
    if (source_image_.empty())
        return;

    calcOpticalFlowPyrLK(source_image_, test_image, source_points_, target_points, status, err);

    // Putting the results into the output structure
    for (int i=0; i<status.size(); ++i){
        if (status[i] == 1){
            output_found_points.push_back(target_points[i]);
            training_point_indices.push_back(i);
        }
    }
}
