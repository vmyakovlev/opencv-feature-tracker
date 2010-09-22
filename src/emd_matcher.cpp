/* This is the contributed code:

Original Version: 2010-09-21  Dat Chu dattanchu@gmail.com
Original Comments:

A descriptor matcher using Earth Mover Distance (EMD) method

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

#include "emd_matcher.h"
#include "blob_feature.h"

using namespace cv;

template<class DistanceType>
void EMDDescriptorMatcher<DistanceType>::index(const std::vector<cv::KeyPoint>& db_keypoints,
                                 const cv::Mat& db_descriptors){
    // copy keypoints and descriptors
    db_keypoints_.assign(db_keypoints.begin(), db_keypoints.end());
    db_descriptors_ = db_descriptors.clone();
}

/** \brief Match the input descriptors

  Make sure that you set the weights with SetWeights() before you call this method.

  \param query_keypoints Keypoints to match
  \param query_descriptors Descriptors to match
  \param[out] matches Returning the indexes of the matches
  \param[out] distance Corresponding matching distance (lower is better)

  \todo Improve efficiency so we don't have to re-allocate signature1_, signature2_ every time

  \see cvCalcEMD2, SetWeights
*/
template<class DistanceType>
void EMDDescriptorMatcher<DistanceType>::match(const std::vector<cv::KeyPoint>& query_keypoints,
                                 const cv::Mat& query_descriptors,
                                 std::vector<int>& matches,
                                 std::vector<float>& distances) const
{
    /* REMINDER:
        Signature matrix follows the description in cvCalcEMD2 where each row contains
        a weight follows by a feature point
    */
    // construct signature1
    signature1_ = Mat(db_descriptors_.rows, db_descriptors_.cols + 1, CV_32F);
    signature2_ = Mat(query_descriptors.rows, query_descriptors.cols + 1, CV_32F);

    // Get access to the column containing the weights
    Mat signature1_weight_col = signature1_.col(0);
    Mat signature2_weight_col = signature2_.col(0);

    // copy weight data over
    weight1_.copyTo(signature1_weight_col);
    weight2_.copyTo(signature2_weight_col);

    // copy descriptor data over
    Mat signature1_descriptor_cols = signature1_.colRange(Range(1,db_descriptors_.cols));
    Mat signature2_descriptor_cols = signature2_.colRange(Range(1,query_descriptors.cols));
    db_descriptors_.copyTo(signature1_descriptor_cols);
    query_descriptors.copyTo(signature2_descriptor_cols);

    // allocate space for flow matrix
    Mat flow_mat(db_descriptors_.rows, query_descriptors.rows, CV_32F);

    double diagonal_image_length = 1000;//cv::sqrt(720*720 + 576*576);
    // time to solve our emd
    cvCalcEMD2(&CvMat(signature1_),
               &CvMat(signature2_),
               //CV_DIST_L1,
               CV_DIST_USER,
               blob_distance,
               NULL,
               &CvMat(flow_mat),
               NULL,
               &diagonal_image_length);

    //cvSave("flow_mat.cvmat",&CvMat(flow_mat));

    // copy data out
    matches.resize(flow_mat.rows);
    distances.resize(flow_mat.rows);

    double min_flow_value;
    double max_flow_value;
    Point max_flow_location;
    Mat current_row;
    Scalar current_row_sum;
    for (int i=0; i<flow_mat.rows; i++){
        current_row = flow_mat.row(i);

        // a match corresponds to the strongest flow
        minMaxLoc(current_row, &min_flow_value, &max_flow_value, 0, &max_flow_location);

        matches[i]  = max_flow_location.x;
        distances[i] = max_flow_value;
    }
    return;
}

/** \brief Set the weights to be used for these descriptors

  \param weight1 a vector of weight (Nx1)
  \param weight2 a vector of weight (Mx1)
*/
template<class DistanceType>
void EMDDescriptorMatcher<DistanceType>::SetWeights(const Mat & weight1, const Mat & weight2){
    weight1_ = weight1;
    weight2_ = weight2;
}
