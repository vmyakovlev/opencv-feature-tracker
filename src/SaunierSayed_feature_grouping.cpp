#include "SaunierSayed_feature_grouping.h"
#include <iostream>

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
        TracksConnectionGraph::edge_descriptor e;
        bool operation_success;

        v = vertex(id, tracks_connection_graph_);
        tracks_connection_graph_[v].activated = true;

        // *****************************************
        // find all close-by tracks and connect them

        // Find nearby tracks: this is naive algorithm but since we didn't build
        // a kd-tree index for these vertex positions, there is not much else we can do
        TracksConnectionGraph::vertex_iterator vi, vi_end;
        float distance;
        for( tie(vi,vi_end) = vertices(tracks_connection_graph_); vi!=vi_end; vi++){
            distance = Distance(v, *vi);
            if (distance < maximum_distance_threshold_){
                // make a connection between these two
                tie(e, operation_success) = add_edge(v, *vi, tracks_connection_graph_);

                if (!operation_success){
                    // TODO: Devise a better warning technique
                    //       Consider exception
                    std::cerr << "Unable to add an edge between " << v << " and " << *vi << std::endl;
                }

                // Assign some initial values for this edge information
                tracks_connection_graph_[e].min_distance = distance;
                tracks_connection_graph_[e].max_distance = distance;
            }
        }
    }

    /**
      \note This method makes copy of the internal data and is thus very slow
    */
    Tracks & TrackManager::tracks(){
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

    float TrackManager::Distance(const TracksConnectionGraph::vertex_descriptor & v1, const TracksConnectionGraph::vertex_descriptor & v2){
        // L2 distance between two tracks' positions
        float norm_l2;

        const cv::Point2f point1 = tracks_connection_graph_[v1].pos;
        const cv::Point2f point2 = tracks_connection_graph_[v2].pos;

        return sqrt( (point1.x - point2.x) * (point1.x - point2.x) +
                     (point1.y - point2.y) * (point1.y - point2.y)
                   );
    }
}


