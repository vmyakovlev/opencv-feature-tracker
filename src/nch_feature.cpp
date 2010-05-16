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
    CV_Error(CV_StsNotImplemented, "This descriptor does not support compute()");
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

    /* NOTE: Check Revision 339 for some code that was under #if 0
       @MF: if the code is useful but not current used, please put its purpose here
    */

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
