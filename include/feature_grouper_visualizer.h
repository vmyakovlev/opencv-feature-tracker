#ifndef __FEATURE_GROUPER_VISUALIZER
#define __FEATURE_GROUPER_VISUALIZER

#include <cv.h>
#include <highgui.h>
#include <string>

#include "misc.h"
#include "SaunierSayed_feature_grouping.h"

using std::string;

using namespace cv;

namespace SaunierSayed{
    class FeatureGrouperVisualizer {
    public:
        FeatureGrouperVisualizer(Mat homography_matrix, SaunierSayed::TrackManager *feature_grouper);
        void NewFrame(Mat new_frame);
        void Draw();
    private:
        SaunierSayed::TrackManager * feature_grouper_;
        string window_;
        Mat homography_matrix_;
        Mat image_;
        VideoWriter video_writer_;
    };
}

#endif
