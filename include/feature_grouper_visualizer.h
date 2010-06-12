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
        void ActivateDrawToFile(const cv::Size2i & frame_size, const string & output_filename = "visualizer.avi", int fourcc = CV_FOURCC('M','J','P','G'));
        void NewFrame(Mat new_frame);
        void Draw();
    private:
        SaunierSayed::TrackManager * feature_grouper_;
        string window_;
        Mat homography_matrix_;
        Mat image_;
        VideoWriter video_writer_;
        bool writing_video_out_;
    };
}

#endif
