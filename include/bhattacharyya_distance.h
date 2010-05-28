#ifndef __BHATTACHARYYA_DISTANCE_H
#define __BHATTACHARYYA_DISTANCE_H

#include <cv.h>
#include "feature.h"

class BhattacharyyaDistance : public Distance
{
public:
	float operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2);
	float operator()(const cv::MatND& descriptor1, const cv::MatND& descriptor2);
};

#endif