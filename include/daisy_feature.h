#ifndef __DAISY_FEATURE_H
#define __DAISY_FEATURE_H
#include <cv.h>
#include "feature.h"
#include <daisy/daisy.h>

/** \class DaisyFeatureExtractor
  Extract DAISY features from input image
 */
class DaisyDescriptorExtractor : public ::DescriptorExtractor
{
public:   
    DaisyDescriptorExtractor(double rad=16, int radq=3, int histq=8, int thq=8);
    ~DaisyDescriptorExtractor();
    void compute(const cv::Mat& image,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors) const;
    void compute(const cv::Mat& image,
                 cv::Mat& query_points,
                 cv::Mat& descriptors) const;
    void compute_dense(const cv::Mat& image,
                         cv::Mat& descriptors) const;
    int feature_length() const;
private:
    int verbose_level_; //!< How verbose is our daisy computation output

    double rad_;
    int radq_;
    int histq_;
    int thq_;

    daisy* desc_; //!< Daisy descriptor structure
    cv::Mat image_; //!< in case the input image is not continous, we need to copy it before processing
};

#endif
