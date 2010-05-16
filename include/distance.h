#ifndef __DISTANCE_H_
#define __DISTANCE_H_

#include <cv.h>

/** \class Distance

  Typically, one uses an implementation of this interface in order to compute
  distance between two descriptors.

  The standard descriptor input are 1xN in size. That is, the descriptor is a
  horizontal vector. This is because it fits into how we keep our descriptors
  in memory after getting them from a DescriptorExtractor.

  The operator() method can be made static in this interface but that will not
  allow implementations to accessing member variables. A parametric distance
  method might requires such capability.

  Your implementation should always support the standard usage:

    DescriptorExtractor descriptor_extractor;
    Mat descriptors1, descriptors2;

    descriptor_extractor.compute(input_image, descriptors1);
    descriptor_extractor.compute(input_image2, descriptors2);

    Distance distance;
    for (int i=0; i<descriptors1.rows; ++i){
        std::cout << "Distance " << distance(descriptors1.row(i), descriptors2.row(i))
                  << std::endl;
    }

  As you can see, the input descriptor is 1xN in size. However, we leave the size
  enforcement to your implementation.

  In some cases, individual distance between two descriptors are combined into
  another distance. This is called distance fusion and is provided by a DistanceFuser.

  */
class Distance
{
    virtual float operator()(const cv::Mat& descriptor1, const cv::Mat& descriptor2) = 0;

    /** \brief Compute distance for high dimensional matrix (aka tensor)

      \note Not supported by default
    */
    virtual float operator()(const cv::MatND& descriptor1, const cv::MatND& descriptor2){
        CV_Error(CV_StsNotImplemented, "This distance do not support MatND");
        return 0;
    }
};

class L1Distance : public Distance
{
    float operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2){
        return cv::norm(descriptor1 - descriptor2, cv::NORM_L1);
    }

    float operator()(const cv::MatND& descriptor1, const cv::MatND & descriptor2){
        return cv::norm(descriptor1 - descriptor2, cv::NORM_L1);
    }
};

class L2Distance : public Distance
{
public:
    float operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2){
        return cv::norm(descriptor1 - descriptor2, cv::NORM_L2);
    }

    float operator()(const cv::MatND& descriptor1, const cv::MatND& descriptor2){
        return cv::norm(descriptor1 - descriptor2, cv::NORM_L2);
    }
};

class BhattacharyyaDistance : public Distance
{
public:
    float operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2);
    float operator()(const cv::MatND& descriptor1, const cv::MatND& descriptor2);
};

class PearsonCoefficientDistance: public Distance
{
public:
    float operator()(const cv::Mat& descriptor1, const cv::Mat & descriptor2);
    float operator()(const cv::MatND& descriptor1, const cv::MatND& descriptor2);
};

#endif
