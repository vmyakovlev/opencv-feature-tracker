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
public:
    virtual void detect(const cv::Mat& image,
                        std::vector<cv::KeyPoint>& keypoints,
                        const cv::Mat& mask = cv::Mat() ) = 0 ;
};

/** \class DescriptorExtractor

  A virtual interface for descriptor extractors. Current implementation is gear
  toward point-based descriptor extractors. For region-based descriptor extractors,
  there is no concept of keypoints.

  \todo Consider splitting into PointDescriptorExtractor and RegionDescriptorExtractor
*/
class DescriptorExtractor
{
    /** \brief Compute sparse descriptor

      \param image Input image
      \param keypoints The interest points where to compute the descriptors
      \param[out] descriptors Output descriptors
    */
    virtual void compute(const cv::Mat& image,
                         std::vector<cv::KeyPoint>& keypoints,
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

/** \class DescriptorMatcher
    \brief Abstract base class for matching

    Make sure your implementation templatize for different Distance(s)
*/
class DescriptorMatcher
{
    /** \brief Index input keypoints (applicable to stuff like ANN)

      Your implementation will likely need to save the input keypoints
      and descriptors since it will likely be used again in matching.

      \param db_keypoints Input database keypoints
      \param db_descriptors Input database descriptors
      */
    virtual void index(const std::vector<cv::KeyPoint>& db_keypoints,
                       const cv::Mat& db_descriptors) = 0;

    /** \brief Find the matches in our database for the input keypoints

    \param query_keypoints Keypoints to pay attention to
    \param query_descriptors Corresponding descriptors for the keypoints
    \param matches Indices of the matches in the database keypoints
    \param distance Distance of each match
    */
    virtual void match(const std::vector<cv::KeyPoint>& query_keypoints,
                       const cv::Mat& query_descriptors,
                       std::vector<int>& matches,
                       std::vector<float>& distance) const = 0;

};

//typedef std::vector<cv::KeyPoint> KeyPointCollection;
/** \class DescriptorMatchGeneric

  A generic descriptor matcher that incorporates both extraction and matching.
  */
class DescriptorMatchGeneric
{
public:

    //! Adds keypoints from several images to the training set (descriptors are supposed to be calculated here)
    //virtual void add(KeyPointCollection& keypoints);

    //! Adds keypoints from a single image to the training set (descriptors are supposed to be calculated here)
    virtual void add(const Mat& image, vector<KeyPoint>& points) = 0;

    //! Classifies test keypoints
    virtual void classify(const Mat& image, vector<KeyPoint>& points){};

    //! Matches test keypoints to the training set
    virtual void match(const Mat& image, vector<KeyPoint>& points, vector<int>& indices){};

    //! Search for training keypoints in the test image
    virtual void search(const Mat& test_image, vector<KeyPoint>& output_found_points, vector<int>& training_point_indices){};
};



#endif
