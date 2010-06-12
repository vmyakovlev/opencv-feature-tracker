#ifndef __NCH_FEATURE_H
#define __NCH_FEATURE_H
#include <cv.h>
#include "feature.h"

/** \class NCHFeatureExtractor
	Extract Normalized Color Histogram as features from input image
 */
class NCHDescriptorExtractor : public DescriptorExtractor
{
public:
	NCHDescriptorExtractor();
	NCHDescriptorExtractor(int bins);
	~NCHDescriptorExtractor();
	void compute(const cv::Mat& image,
                                 std::vector<cv::KeyPoint>& keypoints,
				 cv::Mat& descriptors);
	void compute_dense(const cv::Mat& image,
						cv::MatND& descriptor);
	void compute_dense(const std::vector<cv::Mat>& images,
						std::vector<cv::MatND>& descriptors);
	int get_bins();
private:
	int bins_;
};

#endif
