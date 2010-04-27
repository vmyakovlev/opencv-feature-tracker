#ifndef __EMD_MATCHER_H
#define __EMD_MATCHER_H

#include <cv.h>
#include "feature.h"
#include <exception>

/** \class EMDDescriptorMatcher
    \todo Make distance option settable
    \todo Make diagonal image length settable
  */
class EMDDescriptorMatcher : public DescriptorMatcher
{
public:
    // mask is an MxN matrix encoding which pair-wise distances should be computed, so we can use James' SSE approach for masking out irrelevant pairs.
    void match(const cv::Mat& descriptors_1,
                       const cv::Mat& descriptors_2,
                       IndexPairs& matching_indexes,
                       std::vector<float> & matching_strength,
                       const cv::Mat& mask = cv::Mat() );
    void SetWeights(const Mat & weight1, const Mat & weight2);
private:
    Mat signature1_; //!< Signature matrix for descriptor 1
    Mat signature2_; //!< Signature matrix for descriptor 2
    Mat weight1_;
    Mat weight2_;
};

#endif
