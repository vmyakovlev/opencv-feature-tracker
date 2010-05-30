#include "SaunierSayed_feature_grouping.h"

namespace SaunierSayed{
    TrackManager::TrackManager(int min_num_frame_tracked, float maximum_distance_threshold, float feature_segmentation_threshold){
        min_num_frame_tracked_ = min_num_frame_tracked;
        maximum_distance_threshold_ = maximum_distance_threshold;
        feature_segmentation_threshold_ = feature_segmentation_threshold;
    }

    void TrackManager::AddPoints(const std::vector<cv::Point2f> & new_points){
        TracksConnectionGraph::vertex_descriptor v;

        for (int i=0; i<new_points.size(); ++i){
            // add this new point as a new track in our graph
            v = add_vertex(tracks_connection_graph_);

            // set default property values for this vertex
            tracks_connection_graph_[v].pos = new_points[i];
            tracks_connection_graph_[v].activated = false;
            tracks_connection_graph_[v].number_of_times_tracked = 1;
        }
    }

    void TrackManager::UpdatePoints(const std::vector<cv::Point2f> & new_points, const std::vector<int> & old_points_indices){
        TracksConnectionGraph::vertex_descriptor v;

        for (int i=0; i<old_points_indices.size(); i++){
            v = vertex(i, tracks_connection_graph_);

            tracks_connection_graph_[v].pos = new_points[i];
            tracks_connection_graph_[v].number_of_times_tracked++;

            // NOTE: activation only happens once
            if (tracks_connection_graph_[v].number_of_times_tracked == min_num_frame_tracked_){
                ActivateTrack(i);
            }
        }
    }

    void TrackManager::ActivateTrack(int id){
        TracksConnectionGraph::vertex_descriptor v;

        v = vertex(id, tracks_connection_graph_);
        tracks_connection_graph_[v].activated = true;

        // find all close-by tracks and connect them
    }

    /**
      \note This method makes copy of the internal data and is thus very slow
    */
    Tracks TrackManager::tracks(){
        Tracks all_track_information;

        TracksConnectionGraph::vertex_descriptor v;
        TracksConnectionGraph::vertices_size_type i;
        for (i=0; i<num_vertices(tracks_connection_graph_); i++){
            v = vertex(i, tracks_connection_graph_);
            all_track_information[i].pos = tracks_connection_graph_[v].pos;
            all_track_information[i].number_of_times_tracked = tracks_connection_graph_[v].number_of_times_tracked;
            all_track_information[i].id = tracks_connection_graph_[v].id;
            all_track_information[i].activated = tracks_connection_graph_[v].activated;
        }

        return all_track_information;
    }

    int TrackManager::num_tracks(){
        return num_vertices(tracks_connection_graph_);
    }

    void TrackManager::AddPossiblyDuplicatePoints(const std::vector<cv::Point2f> & new_points){
        std::vector<cv::Point2f> copy_points = new_points;
        RemoveDuplicatePoints(copy_points);
        AddPoints(copy_points);
    }

    //! Remove points that is already in one of the tracks
    void TrackManager::RemoveDuplicatePoints(std::vector<cv::Point2f> & input_points){
        std::vector<cv::Point2f> cleaned_input_points;
        std::vector<cv::Point2f>::iterator it;

        // if the point positions are indexed, this search could be a bit better
        TracksConnectionGraph::vertex_iterator v, vend;

        float norm_l1;
        bool to_be_removed;
        cv::Point2f temp_point;

        for (it=input_points.begin(); it!=input_points.end(); ++it){
            to_be_removed = false;

            for (tie(v,vend) = vertices(tracks_connection_graph_); v!=vend; ++v){
                temp_point = tracks_connection_graph_[*v].pos;

                norm_l1 = abs( temp_point.x - (*it).x +
                                temp_point.y - (*it).y );

                if ( norm_l1 < 2){
                    to_be_removed = true;
                    break;
                }
            }

            if (!to_be_removed)
                cleaned_input_points.push_back(*it);
        }

        input_points = cleaned_input_points;
    }
}
