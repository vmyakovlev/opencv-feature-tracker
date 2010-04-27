#include "emd_matcher.h"
#include "blob_feature.h"

/** \brief Match two descriptor using EMD matcher

  Make sure that you set the weights with SetWeights() before you call this method.

  \param descriptors_1 Descriptor extracted (e.g. from BlobDescriptorExtractor)
  \param descriptors_2 Descriptor extracted
  \param[out] matching_indexes Returning the indexes of the matching pair
  \param[out] matching_strength Corresponding matching strength

  \todo Incorporate mask
  \todo Improve efficiency so we don't have to re-allocate signature1_, signature2_ every time

  \see cvCalcEMD2, SetWeights
*/
void EMDDescriptorMatcher::match(const cv::Mat& descriptors_1,
                                 const cv::Mat& descriptors_2,
                                 IndexPairs& matching_indexes,
                                 std::vector<float> & matching_strength,
                                 const cv::Mat& mask ){
    /* REMINDER:
        Signature matrix follows the description in cvCalcEMD2 where each row contains
        a weight follows by a feature point
    */
    // construct signature1
    signature1_ = Mat(descriptors_1.rows, descriptors_1.cols + 1, CV_32F);
    signature2_ = Mat(descriptors_2.rows, descriptors_2.cols + 1, CV_32F);

    // awesome
    Mat signature1_weight_col = signature1_.col(0);
    Mat signature2_weight_col = signature2_.col(0);

    // copy weight data over
    weight1_.copyTo(signature1_weight_col);
    weight2_.copyTo(signature2_weight_col);

    // copy descriptor data over
    Mat signature1_descriptor_cols = signature1_.colRange(Range(1,descriptors_1.cols));
    Mat signature2_descriptor_cols = signature2_.colRange(Range(1,descriptors_2.cols));
    descriptors_1.copyTo(signature1_descriptor_cols);
    descriptors_2.copyTo(signature2_descriptor_cols);

    // allocate space for flow matrix
    Mat flow_mat(descriptors_1.rows, descriptors_2.rows, CV_32F);

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
    matching_indexes.resize(flow_mat.rows);
    matching_strength.resize(flow_mat.rows);

    double min_flow_value;
    double max_flow_value;
    Point max_flow_location;
    Mat current_row;
    Scalar current_row_sum;
    for (int i=0; i<flow_mat.rows; i++){
        current_row = flow_mat.row(i);

        // a match corresponds to the strongest flow
        minMaxLoc(current_row, &min_flow_value, &max_flow_value, 0, &max_flow_location);

        matching_indexes[i].first = i;
        matching_indexes[i].second = max_flow_location.x;

        // matching strength = flow percentage of the strongest flow
        current_row_sum = sum(current_row);
        matching_strength[i] = max_flow_value / current_row_sum[0];
    }
    return;
}

/** \brief Set the weights to be used for these descripts

  \param weight1 a vector of weight (Nx1)
  \param weight2 a vector of weight (Mx1)
*/
void EMDDescriptorMatcher::SetWeights(const Mat & weight1, const Mat & weight2){
    weight1_ = weight1;
    weight2_ = weight2;
}
