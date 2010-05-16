#include <cv.h>
#include "feature.h"
using namespace cv;

void DescriptorExtractor::compute_dense(const cv::Mat& image, cv::Mat& descriptors){
    CV_Error(CV_StsNotImplemented, "This descriptor do not have dense computation implemented");
}
