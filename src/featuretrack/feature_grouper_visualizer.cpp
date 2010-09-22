/* This is the contributed code:

Original Version: 2010-09-21  Dat Chu dattanchu@gmail.com
Original Comments:

A visualizer that can be used to visualize to a video the results of your tracker
at every frame.clone See FeatureBasedTracking.cxx for an example.

*/

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

#include "feature_grouper_visualizer.h"
#include "color_pallete.h"

namespace SaunierSayed{
    FeatureGrouperVisualizer::FeatureGrouperVisualizer(Mat homography_matrix, SaunierSayed::TrackManager *feature_grouper){
        window_ = "Feature Grouper status";
        homography_matrix_ = homography_matrix;
        feature_grouper_ = feature_grouper;
        writing_video_out_ = false;
        is_draw_coordinate = false;

        // Create window
        namedWindow(window_);
    }

    void FeatureGrouperVisualizer::NewFrame(Mat new_frame){
        image_ = new_frame.clone();
    }

    void FeatureGrouperVisualizer::ActivateDrawToFile(const cv::Size2i &output_video_frame_size, const string & output_filename, int fourcc){
        bool video_writer_success;
        if (!video_writer_.isOpened()){
            video_writer_success = video_writer_.open(output_filename, fourcc, 30, output_video_frame_size);

            if (!video_writer_success){
                std::cerr << "Error: Unable to create video writer with supplied arguments\n";
            }
        }
        else
            std::cout << "Warning: Video writer already activated. Ignoring re-activation request.\n";

        writing_video_out_ = true;
    }

    /** \brief Draw track information on the image

      Go through each vertex. Draw it as a circle.
      For each adjacent vertex to it. Draw a line to it.

      */
    void FeatureGrouperVisualizer::Draw(){
        TracksConnectionGraph::vertex_iterator vi, viend;
        TracksConnectionGraph::out_edge_iterator ei, eiend;
        TracksConnectionGraph::vertex_descriptor other_v;

        TracksConnectionGraph & graph = feature_grouper_->tracks_connection_graph_;

        cv::Point2f position_in_image,
            position_in_image2,
            position_in_world,
            position_to_draw;
        CvScalar color;

        char position_text[256];
        for (tie(vi, viend) = vertices(graph); vi != viend; ++vi ){
            // Convert position to image coordinate
            position_in_world = (graph)[*vi].pos;
            convert_to_image_coordinate(position_in_world, homography_matrix_, &position_in_image);

            TracksConnectionGraph::vertices_size_type num_components;
//            std::vector<TracksConnectionGraph::vertices_size_type> connected_components_map = feature_grouper_->GetConnectedComponentsMap(num_components);

            if (is_draw_inactive){
                // Draw this track with only two colors (blue for those tracked for a long time, red otherwise)
                if(graph[*vi].previous_displacements.size() >= feature_grouper_->maximum_previous_points_remembered_){
                    color = CV_RGB(0,0,255);
                } else {
                    color = CV_RGB(255,0,0);
                }
                circle(image_, position_in_image, 1, color);
            } else {
                if (graph[*vi].activated){
                    // color this vertex based on its assigned component_id
                    color = ColorPallete::colors[graph[*vi].component_id % ColorPallete::NUM_COLORS_IN_PALLETE];

                    circle(image_, position_in_image, 1, color);
                }
            }

            // Write Text Information for this track
            if (is_draw_coordinate){
                sprintf(position_text, "%d(%5.1f,%5.1f)", (graph)[*vi].id, position_in_world.x, position_in_world.y);
                position_to_draw.x = position_in_image.x + 5;
                position_to_draw.y = position_in_image.y + 5;
                putText(image_, position_text, position_to_draw, FONT_HERSHEY_PLAIN, 0.4, CV_RGB(128,128,0));
            }

            // Draw lines to adjacent vertices (if the edge is active)
            for (tie(ei, eiend) = out_edges(*vi, graph); ei!=eiend; ++ei){
                if (!graph[*ei].active)
                    continue;

                // Get where this out_edge is pointing to
                other_v = target(*ei, graph);

                // Convert position to image coordinate
                position_in_world = (graph)[other_v].pos;
                convert_to_image_coordinate(position_in_world, homography_matrix_, &position_in_image2);

                line(image_, position_in_image, position_in_image2, color);
            }
        }
    }

    void FeatureGrouperVisualizer::ShowAndWrite(){
        imshow(window_, image_);
        if (writing_video_out_)
            video_writer_ << image_;
    }

    void FeatureGrouperVisualizer::CustomDraw(const std::vector<Point2f> & points, CvScalar color /*= CV_RGB(255,255,0)*/, bool is_required_homography_transform){
        cv::Point2f position_in_image;

        for (int i=0; i<points.size(); ++i){
            if (is_required_homography_transform){
                convert_to_image_coordinate(points[i], homography_matrix_, &position_in_image);
            } else {
                position_in_image = points[i];
            }

            circle(image_, position_in_image, 1, color);
        }
    }
}
