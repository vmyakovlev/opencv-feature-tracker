#ifndef __DESCRIPTOR_MATCH_H
#define __DESCRIPTOR_MATCH_H

#include "feature.h"
#include <vector>
#include <cv.h>

class KLTTracker : public DescriptorMatchGeneric {
public:
    //! Adds keypoints from a single image to the training set (descriptors are supposed to be calculated here)
    virtual void add(const Mat& image, vector<KeyPoint>& points);

    //! Classifies test keypoints
    /**
      Does nothing in KLT tracker
      */
    virtual void classify(const Mat& image, vector<KeyPoint>& points){
        CV_Error(CV_StsNotImplemented, "This matcher doesn't support classifying. Use searching instead.");
    }

    //! Matches test keypoints to the training set
    virtual void match(const Mat& image, vector<KeyPoint>& points, vector<int>& indices){
        CV_Error(CV_StsNotImplemented, "This matcher doesn't support matching. Use searching instead.");
    }

    /**
        \see DescriptorMatchGeneric
    */
    virtual void search(const Mat& test_image, vector<Point2f>& output_found_points, vector<int>& training_point_indices);
private:
    Mat source_image_;
    std::vector<cv::Point2f> source_points_;
};

#endif
