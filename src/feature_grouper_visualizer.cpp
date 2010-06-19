#include "feature_grouper_visualizer.h"

namespace SaunierSayed{
    FeatureGrouperVisualizer::FeatureGrouperVisualizer(Mat homography_matrix, SaunierSayed::TrackManager *feature_grouper){
        window_ = "Feature Grouper status";
        homography_matrix_ = homography_matrix;
        feature_grouper_ = feature_grouper;
        writing_video_out_ = false;

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
        TracksConnectionGraph::adjacency_iterator vi2, vi2end;

        TracksConnectionGraph & graph = feature_grouper_->tracks_connection_graph_;

        cv::Point2f position_in_image,
            position_in_image2,
            position_in_world;

        for (tie(vi, viend) = vertices(graph); vi != viend; ++vi ){
            // Convert position to image coordinate
            position_in_world = (graph)[*vi].pos;
            convert_to_image_coordinate(position_in_world, homography_matrix_, &position_in_image);

            // Draw this track
            circle(image_, position_in_image, 1, CV_RGB(255,0,0));

            // Draw lines to adjacent vertices
            for (tie(vi2, vi2end)=adjacent_vertices(*vi, graph); vi2!=vi2end; ++vi2){
                // Convert position to image coordinate
                position_in_world = (graph)[*vi2].pos;
                convert_to_image_coordinate(position_in_world, homography_matrix_, &position_in_image2);

                line(image_, position_in_image, position_in_image2, CV_RGB(0,255,0));
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
