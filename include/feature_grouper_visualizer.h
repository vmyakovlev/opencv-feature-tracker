#ifndef __FEATURE_GROUPER_VISUALIZER
#define __FEATURE_GROUPER_VISUALIZER

#include <cv.h>
#include <highgui.h>
#include <string>
#include <vector>

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

        /** \brief Perform custom drawing of points on the image

          Use this method to provide some custom visualization of certain points

          \param new_points Points to draw
          \param color The color of the circle
          \param is_required_homography_transform If yes, these points will be transformed with the internally saved homography matrix before being drawn
        */
        void CustomDraw(const std::vector<Point2f> & new_points, CvScalar color = CV_RGB(255,255,0), bool is_required_homography_transform = false);

        /** \brief Show the image and draw the frame to image
        */
        void ShowAndWrite();

        // public parameters
        bool is_draw_coordinate;
        bool is_draw_inactive;
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
