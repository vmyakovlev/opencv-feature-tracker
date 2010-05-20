#ifndef __FEATURE_DETECTOR_H_
#define __FEATURE_DETECTOR_H_

#include "cv.h"
#include "feature.h"

class ShiTomashiFeatureDetector : public FeatureDetector{
public:
    ShiTomashiFeatureDetector(int max_corners = 100, double quality_level = 0.1, double min_distance = 5,
                              int block_size = 3, bool use_harris_detector = false, double k = 0.04) :
            max_corners_(max_corners), quality_level_(quality_level), min_distance_(min_distance),
            block_size_(block_size), use_harris_detector_(use_harris_detector), k_(k){
    }

    virtual void detect(const cv::Mat& image,
                        std::vector<cv::KeyPoint>& keypoints,
                        const cv::Mat& mask = cv::Mat() ){

        vector<Point2f> corners;
        cv::goodFeaturesToTrack(image, corners, max_corners_, quality_level_, min_distance_, mask, block_size_, use_harris_detector_, k_ );

        // convert these Point2f into cv::KeyPoint
        keypoints.clear();
        keypoints.resize(corners.size());

        for (int i=0; i<keypoints.size(); ++i){
            keypoints[i] = cv::KeyPoint(corners[i], 1);
        }
    }

    int max_corners_;
    double quality_level_;
    double min_distance_;
    int block_size_;
    bool use_harris_detector_;
    double k_;
};

#endif
