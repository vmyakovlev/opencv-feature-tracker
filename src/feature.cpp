#include <cv.h>
#include "feature.h"

void DescriptorExtractor::compute_dense(const cv::Mat& image, cv::Mat& descriptors){
    CV_Error(CV_StsNotImplemented, "This descriptor do not have dense computation implemented");
}
