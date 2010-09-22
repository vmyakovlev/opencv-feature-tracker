/*///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//*/
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
