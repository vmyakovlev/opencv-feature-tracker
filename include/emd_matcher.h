#ifndef __EMD_MATCHER_H
#define __EMD_MATCHER_H

#include <cv.h>
#include "feature.h"
#include <exception>

/** \class EMDDescriptorMatcher
    \todo Make distance option settable
    \todo Make diagonal image length settable
  */
template<class DistanceType>
class EMDDescriptorMatcher : public DescriptorMatcher
{
public:
    EMDDescriptorMatcher(DistanceType distance = DistanceType() );
    virtual void index(const std::vector<cv::KeyPoint>& db_keypoints,
                       const cv::Mat& db_descriptors);
    virtual void match(const std::vector<cv::KeyPoint>& query_keypoints,
                       const cv::Mat& query_descriptors,
                       std::vector<int>& matches,
                       std::vector<float>& distance) const;
    void SetWeights(const Mat & weight1, const Mat & weight2);
private:
    // These signature matrices are to be set up so that
    // cvCalcEMD2 can be used
    Mat signature1_; //!< Signature matrix for descriptor 1
    Mat signature2_; //!< Signature matrix for descriptor 2

    Mat weight1_;
    Mat weight2_;

    // Save these information from index() so they can be used in match()
    std::vector<cv::KeyPoint> db_keypoints_;
    cv::Mat db_descriptors_;
};

#endif
