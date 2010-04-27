#include "nch_feature.h"
#include <iostream>

using std::cout;
using std::endl;

NCHDescriptorExtractor::NCHDescriptorExtractor()
{
	bins_ = 8;
}

NCHDescriptorExtractor::NCHDescriptorExtractor( int bins )
	:bins_(bins)
{}

NCHDescriptorExtractor::~NCHDescriptorExtractor() {}

void NCHDescriptorExtractor::compute( const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors )
{
	cout << "FATAL ERROR: Class NCHFeatureExtractor does not have method compute() implemented." << endl
		 << "Method compute_dense() should be called instead." << endl;
	exit(-1);
}

void NCHDescriptorExtractor::compute_dense( const cv::Mat& image, cv::MatND& descriptor )
{
	// assume the input image is 3-channel
	CV_Assert(image.channels() == 3);
	Mat img;
	if (image.type() != CV_32FC3) {
		image.convertTo(img, CV_32FC3);
	}
	else img = image;

	// prepare the MatND histogram
	const int hist_size[] = {bins_, bins_, bins_};
	
	descriptor.create(3, hist_size, CV_32F);
	descriptor = Scalar(0);

	// accumulate the histogram
	int num_of_valid_pixels = 0;
	int bin_max = bins_ - 1;
	MatConstIterator_<Vec3f> it = img.begin<Vec3f>(),
							 it_end = img.end<Vec3f>();
	for ( ; it != it_end; ++it) {
		
		Vec3f pix;
		pix[0] = (*it)[0];
		pix[1] = (*it)[1];
		pix[2] = (*it)[2];
				
		if (pix[0] == -1 || pix[1] == -1 || pix[2] == -1) continue;
		else num_of_valid_pixels++;
		
		// normalize by removing brightness
		float sum = pix[0] + pix[1] + pix[2];
		if (sum != 0) {
			pix[0] = pix[0] / sum;
			pix[1] = pix[1] / sum;
			pix[2] = pix[2] / sum;
		}
		
		// compute index (range: 0 - bins-1)
		int idx1 = floor(pix[0]*bin_max+0.5);
		int idx2 = floor(pix[1]*bin_max+0.5);
		int idx3 = floor(pix[2]*bin_max+0.5);

		descriptor.at<float>(idx1,idx2,idx3) += 1.0f;
	}
	
	normalize(descriptor, descriptor, 1, 0, NORM_L1);
	
	CV_Assert(fabs(sum(descriptor)[0] - 1) < FLT_EPSILON);

#if 0
	int num_of_channels = image.channels();
	Mat* channels = new Mat[num_of_channels];
	if (num_of_channels > 1)
		split(image, channels);
	else channels[0] = image;

	for (int i = 0; i < num_of_channels; ++i)
		channels[i] = channels[i].reshape(0, 1);
	
	descriptor.create(1, 256*num_of_channels, CV_32F);
	descriptor = Scalar(0);

	int num_of_pixels = image.rows * image.cols;
	int num_of_valid_pixels = 0;
	
	for (int i = 0; i < num_of_pixels; ++i) {
		float val = channels[0].at<float>(0, i);
		
		// no need to examine other channels
		if (val == -1) continue;
		else num_of_valid_pixels++;
		
		for (int j = 0; j < num_of_channels; ++j) {
			val = channels[j].at<float>(0, i);
			descriptor.at<float>(0, j*256+static_cast<int>(val))++;
		}
	}

	// normalize
	descriptor = descriptor / num_of_valid_pixels;
	
	float sum = 0.0f;
	for (int i = 0; i < 3*256; ++i) {
		sum += descriptor.at<float>(0,i);
	}

	descriptor = descriptor.reshape(num_of_channels);
	Mat* descriptor_channels = new Mat[num_of_channels];
	split(descriptor, descriptor_channels);

	for (int i = 0; i < num_of_channels; ++i) {
		float sum = 0.0f;
		for (int j = 0; j < 256; ++j)
			sum += descriptor_channels[i].at<float>(0,j);
		descriptor_channels[i] = descriptor_channels[i] / sum;
	}

	//cout << "After normalization" << endl;
	//for (int i = 0; i < num_of_channels; ++i) {
	//	float sum = 0.0f;
	//	for (int j = 0; j < 256; ++j)
	//		sum += descriptor_channels[i].at<float>(0,j);
	//	cout << sum << endl;
	//}
	merge(descriptor_channels, num_of_channels, descriptor);

	delete [] descriptor_channels;
	delete [] channels;
#endif
}

void NCHDescriptorExtractor::compute_dense( const std::vector<cv::Mat>& images, std::vector<cv::MatND>& descriptors )
{
	int num_of_inputs = images.size();
	for (int i = 0; i < num_of_inputs; ++i) {
		MatND descriptor;
		compute_dense(images[i], descriptor);
		descriptors.push_back(descriptor);
	}
}

int NCHDescriptorExtractor::get_bins()
{
	return bins_;
}