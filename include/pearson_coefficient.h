#ifndef __PEARSON_COEFFICIENT_H
#define __PEARSON_COEFFICIENT_H

#include <cv.h>
#include "feature.h"

class PearsonCoefficient: public Distance
{
public:
	float operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2);
	float operator()(const cv::MatND& descriptor1, const cv::MatND& descriptor2);
};

#endif