#include "distance.h"

// http://www.stat.wmich.edu/s216/book/node122.html
float PearsonCoefficientDistance::operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2 )
{
	CV_Assert(descriptor1.type() == CV_32F);
	CV_Assert(descriptor2.type() == CV_32F);
	int num_of_bins = descriptor1.cols;
	CV_Assert(num_of_bins == descriptor2.cols);

	float sum_XY = 0.0f, sum_X = 0.0f, sum_Y = 0.0f, sum_XS = 0.0f, sum_YS = 0.0f;
	for (int i = 0; i < num_of_bins; ++i) {
		float descr1_elem = descriptor1.at<float>(0, i);
		float descr2_elem = descriptor2.at<float>(0, i);
		sum_XY += descr1_elem * descr2_elem;
		sum_X += descr1_elem;
		sum_Y += descr2_elem;
		sum_XS += pow(descr1_elem, 2.0f);
		sum_YS += pow(descr2_elem, 2.0f);
	}

	float res = (sum_XY - sum_X * sum_Y / num_of_bins) / 
		sqrtf((sum_XS - pow(sum_X, 2.0f) / num_of_bins) * (sum_YS - pow(sum_Y, 2.0f) / num_of_bins));

	// MF: since Pearson Coefficient is a measure of correlation [-1, 1], convert it to distance
	res = 1 - fabs(res);

	return res;
}

float PearsonCoefficientDistance::operator()( const cv::MatND& descriptor1, const cv::MatND& descriptor2 )
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

	float sum_XY = 0.0f, sum_XS = 0.0f, sum_YS = 0.0f;
	float sum_X = sum(descriptor1)[0];
	float sum_Y = sum(descriptor2)[0];
	for (int i = 0; i < dim1; ++i)
		for (int j = 0; j < dim2; ++j)
			for (int k = 0; k < dim3; ++k) {
				float descr1_elem = descriptor1.at<float>(i,j,k);
				float descr2_elem = descriptor2.at<float>(i,j,k);
				sum_XY += descr1_elem * descr2_elem;
				sum_XS += pow(descr1_elem, 2.0f);
				sum_YS += pow(descr2_elem, 2.0f);
			}
	
	float res = (sum_XY - sum_X * sum_Y / num_of_bins) / 
		sqrtf((sum_XS - pow(sum_X, 2.0f) / num_of_bins) * (sum_YS - pow(sum_Y, 2.0f) / num_of_bins));

	// MF: since Pearson Coefficient is a measure of correlation [-1, 1], convert it to distance
	res = 1 - fabs(res);

	return res;
}
