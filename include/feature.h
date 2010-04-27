#ifndef __FEATURE_H
#define __FEATURE_H
#include <cv.h>

using namespace cv;

/** \file Interfaces for feature detector and descriptor extractor

  I prefer these class interface functions to be non-const so that their
  results can be cached.

  The problem with this FeatureDetector version is its reliant on cv::KeyPoint.
  cv::KeyPoint is great for most cases but it doesn't allow blob-like features where
  it is better to use hull information.

  For some information about other possible implementations, please see
  http://pr.willowgarage.com/wiki/PluggableDescriptors#Requirements
*/
class FeatureDetector
{
    virtual void detect(const cv::Mat& image,
                        std::vector<cv::KeyPoint>& keypoints,
                        const cv::Mat& mask = cv::Mat() ) = 0 ;
};

class DescriptorExtractor
{
    /** \brief Compute sparse descriptor

      \param image Input image
      \param keypoints The interest points to compute descriptors for
      \param[out] descriptors Output descriptors
    */
    virtual void compute(const cv::Mat& image,
                         const std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors) = 0;

    /** \brief Compute dense descriptor

      \param image Input image
      \param[out] descriptors The computed descriptors. Each line is one descriptor.
                              To access the descriptor at location (y,x). Get the descriptor
                              at row y*image.cols + x
    */
    virtual void compute_dense(const cv::Mat& image,
                         cv::Mat& descriptors);
};

//! Represent matches as pairs of keypoint/descriptors indexes
typedef std::vector< std::pair<int, int> > IndexPairs;

/** \class DescriptorMatcher
    \brief Abstract base class for matching
*/
class DescriptorMatcher
{
    /** \brief Compute a match between two descriptor vectors

    \param matching_indexes the matching indexes
    \param matching_strength how strong is the corresponding match
    \param mask is an MxN matrix encoding which pair-wise distances should be computed,
                  so we can use James' SSE approach for masking out irrelevant pairs.
    \see IndexPairs
    */
    virtual void match(const cv::Mat& descriptors_1,
                       const cv::Mat& descriptors_2,
                       IndexPairs& matching_indexes,
                       std::vector<float> & matching_strength,
                       const cv::Mat& mask = cv::Mat() ) = 0;
};

class Distance
{
	virtual float operator()(const cv::MatND& descriptor1, const cv::MatND& descriptor2) = 0;
};

#endif
