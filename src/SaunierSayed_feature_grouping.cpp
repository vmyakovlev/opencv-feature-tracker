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
            v = vertex(old_points_indices[i], tracks_connection_graph_);

            tracks_connection_graph_[v].pos = new_points[i];
            tracks_connection_graph_[v].number_of_times_tracked++;
        }

        // Now the points positions have been updated, make sure that min and max distance for their edges are updated too
        TracksConnectionGraph::out_edge_iterator edge_it, edge_it_end;
        TracksConnectionGraph::vertex_descriptor the_other_vertex;
        float new_distance;
        for (int i=0; i<old_points_indices.size(); i++){
            v = vertex(old_points_indices[i], tracks_connection_graph_);

            // Loop through all edges
            for ( tie(edge_it, edge_it_end)=out_edges(v, tracks_connection_graph_); edge_it != edge_it_end; edge_it++){
                the_other_vertex = target(*edge_it, tracks_connection_graph_);

                // Compute new distance
                new_distance = Distance(v, the_other_vertex);

                // Update min and max distance
                if (new_distance < tracks_connection_graph_[*edge_it].min_distance){
                    tracks_connection_graph_[*edge_it].min_distance = new_distance;
                }
                if (new_distance > tracks_connection_graph_[*edge_it].max_distance){
                    tracks_connection_graph_[*edge_it].max_distance = new_distance;
                }

            }

        }

        // NOTE: Activation only happens once
        // NOTE: Activation needs to happen after ALL the positions have been updated
        for (int i=0; i<old_points_indices.size(); i++){
            v = vertex(old_points_indices[i], tracks_connection_graph_);
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
            if (distance < maximum_distance_threshold_ && v!=*vi){
                // make a connection between these two
                tie(e, operation_success) = add_edge(v, *vi, tracks_connection_graph_);

                if (!operation_success){
                    // TODO: Devise a better warning technique
                    //       Consider exception
                    std::cerr << "Warning: Unable to add an edge between " << v << " and " << *vi << std::endl;
                }

                // Assign some initial values for this edge information
                tracks_connection_graph_[e].min_distance = distance;
                tracks_connection_graph_[e].max_distance = distance;
            }
        }
    }

    /**
      \todo Consider API change so that we no longer needs to return these track information this way
      \note This method makes copy of the internal data and is thus very slow
    */
    Tracks TrackManager::tracks() const{
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

    bool TrackManager::get_edge_information(int vertex_id_1, int vertex_id_2, LinkInformation * output_link_information) const{
        TracksConnectionGraph::vertex_descriptor v1 = vertex(vertex_id_1, tracks_connection_graph_);
        TracksConnectionGraph::vertex_descriptor v2 = vertex(vertex_id_2, tracks_connection_graph_);

        bool is_there_an_edge;
        TracksConnectionGraph::edge_descriptor e;

        tie(e, is_there_an_edge) = edge(v1, v2, tracks_connection_graph_);

        if (!is_there_an_edge){
            return false;
        } else {
            output_link_information->id = tracks_connection_graph_[e].id;
            output_link_information->min_distance = tracks_connection_graph_[e].min_distance;
            output_link_information->max_distance = tracks_connection_graph_[e].max_distance;
            return true;
        }
    }

    int TrackManager::num_tracks(){
        return num_vertices(tracks_connection_graph_);
    }

    int TrackManager::num_connections(){
        return num_edges(tracks_connection_graph_);
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
        float norm_l2 = -1;

        const cv::Point2f point1 = tracks_connection_graph_[v1].pos;
        const cv::Point2f point2 = tracks_connection_graph_[v2].pos;

        return sqrt( (point1.x - point2.x) * (point1.x - point2.x) +
                     (point1.y - point2.y) * (point1.y - point2.y)
                   );
    }
}
