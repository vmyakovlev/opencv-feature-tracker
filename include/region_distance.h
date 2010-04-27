#ifndef __REGION_SIMILARITY_H
#define __REGION_SIMILARITY_H

#include <cv.h>
#include <iostream>
#include <stdio.h>
#include <vector>

#include "misc.h"

template <class DistT>
class RegionDistance
{
public:
	float operator()(const std::vector<cv::Mat>& descriptors1, 
					 const std::vector<cv::Mat>& descriptors2)
	{
		int num_of_inputs = descriptors1.size();
		CV_Assert(num_of_inputs == descriptors2.size());

		int channels = descriptors1[0].channels();
		CV_Assert(channels == descriptors2[0].channels());

		Mat *descriptor1 = new Mat[channels];
		Mat *descriptor2 = new Mat[channels];
		Mat_<float> dist_sum(1, channels, 0.0f);

		for (int i = 0; i < num_of_inputs; ++i) {
			split(descriptors1[i], descriptor1);
			split(descriptors2[i], descriptor2);

			// MF: naive way of fusing part scores, i.e., averaging
			for (int j = 0; j < channels; ++j) {
				dist_sum[0][j] += distance_type(descriptor1[j], descriptor2[j]);
			}
		}		
		dist_sum /= num_of_inputs;
	
		float res = 0.0f;
		for (int i = 0; i < channels; ++i) {
			// MF: naive way of fusing channel scores, i.e., averaging
			res += dist_sum[0][i];
			//if (dist_sum[0][i] > res)
			//	res = dist_sum[0][i];
		}
		res /= channels;

		printf("R: %.3f G: %.3f B: %.3f\n",dist_sum[0][0],dist_sum[0][1],dist_sum[0][2]);

		delete [] descriptor1;
		delete [] descriptor2;

		return res;
	};

	float operator()(const std::vector<cv::MatND>& descriptors1, 
		const std::vector<cv::MatND>& descriptors2)
	{
		int num_of_inputs = descriptors1.size();
		CV_Assert(num_of_inputs == descriptors2.size());

		float res = 0.0f;

		for (int i = 0; i < num_of_inputs; ++i) {
			// MF: naive way of fusing part scores, i.e., averaging
			float dist = distance_type(descriptors1[i], descriptors2[i]);
			printf("Part %d : %.3f ", i+1, dist);
			res += dist;
		}		
		res /= num_of_inputs;
		printf("\n");

		return res;
	};

private:
	DistT distance_type;
};

#endif
