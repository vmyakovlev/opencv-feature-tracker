#include <cv.h>
#include "feature.h"
using namespace cv;

void DescriptorExtractor::compute_dense(const cv::Mat& image, cv::Mat& descriptors){
    CV_Error(CV_StsNotImplemented, "This descriptor extractor does not have dense computation implemented");
}

float Distance::operator()(const cv::MatND& descriptor1, const cv::MatND& descriptor2) {
    CV_Error(CV_StsNotImplemented, "This distance does not support cv::MatND");
    return 0;
}
