#include "euclidean_distance.h"

float EuclideanDistance::operator()( const cv::MatND& descriptor1, const cv::MatND& descriptor2 )
{
	CV_Assert(descriptor1.type() == CV_32F);
	CV_Assert(descriptor2.type() == CV_32F);
	int dims = descriptor1.dims;
	CV_Assert(dims == 3);
	CV_Assert(dims == descriptor2.dims);
	int dim1 = descriptor1.size[0];
	int dim2 = descriptor1.size[1];
	int dim3 = descriptor1.size[2];

	float dist = 0.0f;
	for (int i = 0; i < dim1; ++i) 
		for (int j = 0; j < dim2; ++j)
			for (int k = 0; k < dim3 ; ++k) {
				float descr1_elem = descriptor1.at<float>(i,j,k);
				float descr2_elem = descriptor2.at<float>(i,j,k);
				dist += pow(descr1_elem - descr2_elem, 2.0f);
			}

	dist = sqrtf(dist);

	return dist;
}