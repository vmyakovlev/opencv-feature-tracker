#include "distance.h"

// http://www.inf.ed.ac.uk/teaching/courses/av/MATLAB/COLOUR/bhattacharyya.m
float BhattacharyyaDistance::operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2 )
{
	CV_Assert(descriptor1.type() == CV_32F);
	CV_Assert(descriptor2.type() == CV_32F);
	int num_of_bins = descriptor1.cols;
	CV_Assert(num_of_bins == descriptor2.cols);

	float sum1 = 0.0f, sum2 = 0.0f;
	float bcoeff = 0.0f;
	for (int i = 0; i < num_of_bins; ++i) {
		bcoeff = bcoeff + sqrt(descriptor1.at<float>(0,i) * descriptor2.at<float>(0,i));
		sum1 += descriptor1.at<float>(0,i);
		sum2 += descriptor2.at<float>(0,i);
	}

	int sum_int = floor(sum1+0.5);
	CV_Assert(sum_int == floor(sum2+0.5));

	// a hack to overcome numerical error
	bcoeff = bcoeff > sum_int ? sum_int : bcoeff;

	float bdist = sqrt(sum_int - bcoeff);

	return bdist;
}

float BhattacharyyaDistance::operator()( const cv::MatND& descriptor1, const cv::MatND& descriptor2 )
{
	CV_Assert(descriptor1.type() == CV_32F);
	CV_Assert(descriptor2.type() == CV_32F);
	int dims = descriptor1.dims;
	CV_Assert(dims == 3);
	CV_Assert(dims == descriptor2.dims);
	int dim1 = descriptor1.size[0];
	int dim2 = descriptor1.size[1];
	int dim3 = descriptor1.size[2];
	int num_of_bins = dim1 * dim2 * dim3;

	float bcoeff = 0.0f;
	for (int i = 0; i < dim1; ++i) 
		for (int j = 0; j < dim2; ++j)
			for (int k = 0; k < dim3 ; ++k) {
				float descr1_elem = descriptor1.at<float>(i,j,k);
				float descr2_elem = descriptor2.at<float>(i,j,k);
				bcoeff += sqrtf(descr1_elem * descr2_elem);
			}

	float sum1 = sum(descriptor1)[0];
	float sum2 = sum(descriptor2)[0];
	CV_Assert(sum1 == sum2);

	// a hack to overcome numerical error
	bcoeff = bcoeff > sum1 ? sum1 : bcoeff;

	float bdist = sqrt(sum1 - bcoeff);

	return bdist;
}
