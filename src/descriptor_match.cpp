#include "descriptor_match.h"
#include "misc.h"
using namespace cv;

//! Adds keypoints from this source image
/** For KLT Tracker, there is no need to extract any features, so we just save these input for later processing
  in match()
*/
void KLTTracker::add(const Mat& image, vector<KeyPoint>& points){
    source_image_ = image;
    vector_one_to_another(points, source_points_);
}

//! Matches test keypoints to the training set
/**
  This implementation group together feature extraction and matching by using OpenCV calcOpticalFlowPyrLK.
  In order to make this implementation more adapted to the new framework, one needs to dissect calcOpticalFlowPyrLK
  into the feature calculation component and the matching component.

  \param test_image input test image
  \param[out] output_found_points where we found the training points in this image
  \param[out] training_point_indices the corresponding indices of the training points for these found points
              e.g. [3,4,5] means the 1st, 2nd, and 3rd elements in output_found_points correspond to the 4th, 5th and 6th elements in
                           the training points. See add().

*/
void KLTTracker::search(const Mat& test_image, vector<Point2f>& output_found_points, vector<int>& training_point_indices){
    vector<Point2f> target_points;
    vector<uchar> status;
    vector<float> err;

    // TODO: add warning messages when this is encountered
    if (source_image_.empty())
        return;

    calcOpticalFlowPyrLK(source_image_, test_image, source_points_, target_points, status, err);

    // Putting the results into the output structures
    output_found_points.clear();
    training_point_indices.clear();
    for (int i=0; i<status.size(); ++i){
        if (status[i] == 1){
            output_found_points.push_back(target_points[i]);
            training_point_indices.push_back(i);
        }
    }
}
