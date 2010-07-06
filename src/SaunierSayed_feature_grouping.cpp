#include "SaunierSayed_feature_grouping.h"
#include <boost/graph/connected_components.hpp>
#include <boost/property_map.hpp>
#include <iostream>

//#define DEBUG_PRINTOUT

namespace SaunierSayed{
    TrackManager::TrackManager(int min_num_frame_tracked, float min_distance_moved_required,
                               float maximum_distance_threshold, float feature_segmentation_threshold,
                               float minimum_variance_required, float min_distance_between_tracks,
                               bool log_track_to_file){
        min_num_frame_tracked_ = min_num_frame_tracked;
        maximum_distance_threshold_ = maximum_distance_threshold;
        feature_segmentation_threshold_ = feature_segmentation_threshold;
        min_distance_moved_required_ = min_distance_moved_required;

        maximum_previous_points_remembered_ = 30;
        minimum_variance_required_ = minimum_variance_required;
        max_num_frames_not_tracked_allowed_ = 30;

        // NOTE: Find this value from the homography matrix
        min_distance_between_tracks_ = min_distance_between_tracks;

        logging_ = log_track_to_file;

        if (logging_){
            log_file_.open("TrackManager.log", std::ios::out);

            if (!log_file_.is_open()){
                std::cerr << "Error: Cannot open log file" << std::endl;
            }
        }

        current_time_stamp_id_ = 0;
        next_track_id_ = 0;
    }

    TrackManager::~TrackManager(){
        if (logging_)
            log_file_.close();
    }

    TrackManager::TrackManager(const TrackManager & other){
        // Perform assignments on everything except log file
        tracks_connection_graph_ =  other.tracks_connection_graph_ ;
        min_num_frame_tracked_ = other.min_num_frame_tracked_;
        maximum_distance_threshold_ = other.maximum_distance_threshold_;
        feature_segmentation_threshold_ = other.feature_segmentation_threshold_;
        current_time_stamp_id_ = other.current_time_stamp_id_;
        min_distance_moved_required_ = other.min_distance_moved_required_;
        maximum_previous_points_remembered_ = other.maximum_previous_points_remembered_;
        minimum_variance_required_ = other.minimum_variance_required_;
        max_num_frames_not_tracked_allowed_ = other.max_num_frames_not_tracked_allowed_;

        // Disable logging on copied object since the log file object is not copyable
        logging_ = false;
    }

    TrackManager& TrackManager::operator=(const TrackManager & other){
        // Handle self assignment
        if (this == &other)
            return *this;

        // Perform assignments on everything except log file
        tracks_connection_graph_ =  other.tracks_connection_graph_ ;
        min_num_frame_tracked_ = other.min_num_frame_tracked_;
        maximum_distance_threshold_ = other.maximum_distance_threshold_;
        feature_segmentation_threshold_ = other.feature_segmentation_threshold_;
        current_time_stamp_id_ = other.current_time_stamp_id_;
        min_distance_moved_required_ = other.min_distance_moved_required_;
        maximum_previous_points_remembered_ = other.maximum_previous_points_remembered_;
        minimum_variance_required_ = other.minimum_variance_required_;
        max_num_frames_not_tracked_allowed_ = other.max_num_frames_not_tracked_allowed_;

        // Disable logging on copied object since the log file object is not copyable
        logging_ = false;

        return *this;
    }

    void TrackManager::AddPoints(const std::vector<cv::Point2f> & new_points, std::vector<int> * assigned_ids){
        // for convenience, we allocate space for the user if he passes in an empty vector<int>
        if (assigned_ids != NULL && assigned_ids->size() == 0)
            assigned_ids->resize(new_points.size(), -1);

        // sanity check
        CV_Assert(assigned_ids == NULL || new_points.size() == assigned_ids->size());

        TracksConnectionGraph::vertex_descriptor v;
        TracksConnectionGraph::vertex_iterator vi, viend;
        TracksConnectionGraph::edge_descriptor e;
        bool is_edge_addition_success;
        float distance;

        for (int i=0; i<new_points.size(); ++i){
            // if this point has been assigned an id, skip it
            if (assigned_ids != NULL && (*assigned_ids)[i] >= 0){
                continue;
            }

            // get all current vertices
            tie(vi, viend) = vertices(tracks_connection_graph_);

            // add this new point as a new track in our graph
            v = add_vertex(tracks_connection_graph_);

            // set default property values for this vertex
            tracks_connection_graph_[v].id = next_track_id_;
            next_track_id_++;

            tracks_connection_graph_[v].pos = new_points[i];
            tracks_connection_graph_[v].activated = false;
            tracks_connection_graph_[v].number_of_times_tracked = 1;
            tracks_connection_graph_[v].average_position = new_points[i];
            tracks_connection_graph_[v].previous_points.push_back(new_points[i]);
            tracks_connection_graph_[v].last_time_tracked = current_time_stamp_id_;

            // write out the newly assigned id for this point
            if (assigned_ids != NULL){
                (*assigned_ids)[i] = tracks_connection_graph_[v].id;
            }

            // connect this new vertex to all current vertices
            for(; vi!=viend; vi++){
                tie(e, is_edge_addition_success) = add_edge(v, *vi, tracks_connection_graph_);

                // Assign some initial values for this edge information
                distance = Distance(v, *vi);
                tracks_connection_graph_[e].active = false;
                tracks_connection_graph_[e].min_distance = distance;
                tracks_connection_graph_[e].max_distance = distance;
            }
        }
    }
    /** \brief Update our points stored in the graph with new locations

      \param new_points New locations for the points
      \param old_points_ids The corresponding IDs of the old points. This is NOT the internal vertex id but the Track id
    */
    void TrackManager::UpdatePoints(const std::vector<cv::Point2f> & new_points, const std::vector<int> & old_points_ids){
        TracksConnectionGraph::vertex_descriptor v;

        std::vector<TracksConnectionGraph::vertex_descriptor> old_points_vertices = GetVertexDescriptors(old_points_ids);

        //*********************************************************************************************
        // UPDATE TRACKS WITH NEW INFORMATION
        int number_of_times_tracked;
        for (int i=0; i<old_points_ids.size(); i++){
            v = old_points_vertices[i];

            number_of_times_tracked = tracks_connection_graph_[v].number_of_times_tracked;
            tracks_connection_graph_[v].previous_points.push_back(new_points[i]);
            tracks_connection_graph_[v].last_time_tracked = current_time_stamp_id_;

            if (tracks_connection_graph_[v].previous_points.size() > maximum_previous_points_remembered_){
                // update average position
                cv::Point2f old_point = tracks_connection_graph_[v].previous_points.front();
                tracks_connection_graph_[v].previous_points.pop_front();
                tracks_connection_graph_[v].average_position = (tracks_connection_graph_[v].average_position * maximum_previous_points_remembered_ - old_point + new_points[i]) * (1.0 / maximum_previous_points_remembered_);
            } else {
                tracks_connection_graph_[v].average_position = (tracks_connection_graph_[v].average_position * number_of_times_tracked + new_points[i])* (1.0 / (number_of_times_tracked+1));
            }

//            printf("Average position: %f %f\n", tracks_connection_graph_[v].average_position.x, tracks_connection_graph_[v].average_position.y);

            tracks_connection_graph_[v].pos = new_points[i];
            tracks_connection_graph_[v].number_of_times_tracked++;
        }

        //*********************************************************************************************
        // Now the points positions have been updated, make sure that min and max distance for their edges are updated too
        TracksConnectionGraph::out_edge_iterator edge_it, edge_it_end;
        TracksConnectionGraph::vertex_descriptor the_other_vertex;
        float new_distance;
        for (int i=0; i<old_points_ids.size(); i++){
            v = old_points_vertices[i];

            // Loop through all edges
            for ( tie(edge_it, edge_it_end)=out_edges(v, tracks_connection_graph_); edge_it != edge_it_end; edge_it++){
                the_other_vertex = target(*edge_it, tracks_connection_graph_);
                //std::cout << "The other vertex: " << the_other_vertex << std::endl;

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

        std::set<int> vertices_to_remove;

        //*********************************************************************************************
        // REMOVE TRACKS THAT ARE NOT TRACKED FOR AWHILE

        // NOTE: If we find a way to easily sort the old_points_indices (and its respective new_points)
        //       we can avoid having to do a second loop
        TracksConnectionGraph::vertex_iterator vi, viend;
        for (tie(vi, viend)=vertices(tracks_connection_graph_); vi!=viend; ++vi){
            // if we haven't been tracking this point for a while. Time to remove it
            if (tracks_connection_graph_[*vi].last_time_tracked - current_time_stamp_id_ > max_num_frames_not_tracked_allowed_){
                vertices_to_remove.insert(tracks_connection_graph_[*vi].id);
#ifdef DEBUG_PRINOUT
                std::cout << "Track #" << *vi << " removed for not being tracked for some time" << std::endl;
#endif

            }
        }

        //*********************************************************************************************
        // ACTIVATE TRACKS THAT HAVE BEEN TRACKED LONG ENOUGH
        // REMOVE TRACKS THAT HAVE NOT MOVED AROUND FOR A WHILE

        // NOTE: Activation only happens once
        // NOTE: Activation needs to happen after ALL the positions have been updated
        float distance_moved_from_average;
        TracksConnectionGraph::vertex_descriptor vert;
        for (int i=0; i<old_points_ids.size(); i++){
            vert = old_points_vertices[i];

            // TODO: Shouldn't we calculate distance moved from average BEFORE we update the position?
            distance_moved_from_average = Distance(tracks_connection_graph_[vert].average_position, tracks_connection_graph_[vert].pos);
            if (tracks_connection_graph_[vert].activated == false
                && tracks_connection_graph_[vert].number_of_times_tracked >= min_num_frame_tracked_
                && distance_moved_from_average > min_distance_moved_required_)
            {
                ActivateTrack(vert);
                continue; // so that this point is not tested right away whether it has been moved or not
            }

            // Determine for activated points whether there has been enough variance
            // Not enough variance means the track has been stuck for a while
            if (tracks_connection_graph_[vert].activated &&
                tracks_connection_graph_[vert].previous_points.size() >= maximum_previous_points_remembered_){
                // calculate the variance of the previous points
                cv::Point2f variance(0,0);
                cv::Point2f average_pos = tracks_connection_graph_[vert].average_position;
                std::deque<cv::Point2f>::iterator it = tracks_connection_graph_[vert].previous_points.begin(),
                    it_end = tracks_connection_graph_[vert].previous_points.end();
                for (;it!=it_end; ++it){
                    variance.x += (average_pos.x - it->x)*(average_pos.x - it->x);
                    variance.y += (average_pos.y - it->y)*(average_pos.y - it->y);
                }
                variance *= (1.0 / tracks_connection_graph_[vert].previous_points.size());

                // if the variance is too low, time to remove it
                if (variance.x < minimum_variance_required_ && variance.y < minimum_variance_required_){
                    // NOTE: Remember that BGL will try to keep the vertices ID stay in a continuous range
                    //       Thus, we cannot remove these tracks now and expect the old_point_indices to still work correctly
                    //       in the next iteration
#ifdef DEBUG_PRINOUT
                    std::cout << "Track #" << vert << " removed for not moving." << std::endl;
#endif
                    vertices_to_remove.insert(tracks_connection_graph_[vert].id);
                }
            }
        }

        SegmentFarAwayTracks();

        // Now that we no longer use old_points_ids/old_points_vertices content, we can go on and remove these tracks
        std::set<int>::iterator it = vertices_to_remove.begin();
        for (; it != vertices_to_remove.end(); ++it){
            printf("Removing track #%d\n", *it);
            DeleteTrack(*it);
        }


        // Log: current information of each track at this time frame
        if (logging_){
            LogCurrentTrackInfo();
        }



#ifdef DEBUG_PRINTOUT
        // Debug: Report current stats of the graph
        printf("Num vertices: %d\n", num_vertices(tracks_connection_graph_));
        printf("Num edges: %d\n", num_edges(tracks_connection_graph_));
#endif
    }

    void TrackManager::LogCurrentTrackInfo(){
        // iterate through tracks
        TracksConnectionGraph::vertex_iterator vi, viend;
        TracksConnectionGraph::adjacency_iterator vi2, vi2end;

        for (tie(vi, viend) = vertices(tracks_connection_graph_); vi != viend; ++vi ){
            // Write this track information to file
            log_file_ << current_time_stamp_id_ << " "
                    << tracks_connection_graph_[*vi].id << " "
                    << tracks_connection_graph_[*vi].pos.x << " "
                    << tracks_connection_graph_[*vi].pos.y << " "
                    << tracks_connection_graph_[*vi].activated << " ";

            // print the list of vertices connected to this vertex
            for (tie(vi2, vi2end)=adjacent_vertices(*vi, tracks_connection_graph_); vi2!=vi2end; ++vi2){
                log_file_ << tracks_connection_graph_[*vi2].id << " ";
            }

            log_file_ << std::endl;
        }
    }

    void TrackManager::ActivateTrack(TracksConnectionGraph::vertex_descriptor v){
        TracksConnectionGraph::edge_descriptor e;
        bool operation_success;

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
                tie(e, operation_success) = edge(v, *vi, tracks_connection_graph_);

                if (!operation_success){
                    // Although they are close to each other, they have been moving non-harmoniously
                    // and the edge had been previously severed
                    continue;
                }

                tracks_connection_graph_[e].active = true;
            }
        }
    }

    void TrackManager::SegmentFarAwayTracks(){
        // go through all edges
        TracksConnectionGraph::edge_iterator edge_it, edge_it_end, next_it;
        float min_distance, max_distance, distance_range;
        tie(edge_it, edge_it_end)=edges(tracks_connection_graph_);
        for (next_it=edge_it; edge_it!=edge_it_end; edge_it=next_it){
            next_it++;

            // Check that this edge satisfy the condition
            min_distance = tracks_connection_graph_[*edge_it].min_distance;
            max_distance = tracks_connection_graph_[*edge_it].max_distance;
            distance_range = max_distance - min_distance;

            if (distance_range > feature_segmentation_threshold_){
                // severe this edge
                // Care must be taken not to severe the CURRENT iterator, hence the use of next_it
                // SEE: http://www.boost.org/doc/libs/1_43_0/libs/graph/doc/adjacency_list.html
#ifdef DEBUG_PRINTOUT
                TracksConnectionGraph::vertex_descriptor v, v2;
                v = source(*edge_it, tracks_connection_graph_);
                v2 = target(*edge_it, tracks_connection_graph_);
                printf("Edge (%ld,%ld) removed since its vertices move too much: %6.3f (threshold = %f)\n", tracks_connection_graph_[v].id,
                       tracks_connection_graph_[v2].id, distance_range, feature_segmentation_threshold_);
#endif
                remove_edge(*edge_it, tracks_connection_graph_);
            }
        }
    }

    void TrackManager::AdvanceToNextFrame(){
        current_time_stamp_id_++;
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
            all_track_information[i] = get_track_information(v);
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
            output_link_information->active = tracks_connection_graph_[e].active;
            return true;
        }
    }

    int TrackManager::num_tracks(){
        return num_vertices(tracks_connection_graph_);
    }

    int TrackManager::num_connections(){
        return num_edges(tracks_connection_graph_);
    }

    void TrackManager::AddPossiblyDuplicatePoints(const std::vector<cv::Point2f> & new_points, std::vector<int> * assigned_ids){
        FindDuplicatePointIds(new_points, assigned_ids);
        AddPoints(new_points, assigned_ids);
    }

    //! Remove points that are already in one of the tracks
    void TrackManager::RemoveDuplicatePoints(std::vector<cv::Point2f> & input_points){
        std::vector<cv::Point2f> cleaned_input_points;

        // Find the possible duplicates
        std::vector<int> found_ids;
        FindDuplicatePointIds(input_points, &found_ids);

        // For the one that we found ids for, remove them
        for (int i=0; i<found_ids.size(); i++){
            if (found_ids[i] < 0){
                // duplicate didn't find this, copy it
                cleaned_input_points.push_back(input_points[i]);
            }
        }

        input_points = cleaned_input_points;
    }

    /** \brief Find the ids of duplicate points. Return -1 for points that are not a possible duplicate

      Example FindDuplicatePointIds([1,2,3]). Let's assume that 1 and 2 are duplicates and 3 is not.
      The returned assigned_ids will then be: [id_of_1, id_of_2, -1]

      \param new_points New points to test whether they are duplicate
      \param[out] assigned_ids The assigned ids for the duplicate points, -1 for non-duplicate points
      */
    void TrackManager::FindDuplicatePointIds(const std::vector<cv::Point2f> & new_points, std::vector<int> * assigned_ids){
        // for convenience, we allocate space for the user if he passes in an empty vector<int>
        if (assigned_ids != NULL && assigned_ids->size() == 0)
            assigned_ids->resize(new_points.size(), -1);

        // sanity check
        CV_Assert(assigned_ids == NULL || new_points.size() == assigned_ids->size());

        // if the point positions are indexed, this search could be a bit better
        TracksConnectionGraph::vertex_iterator v, vend;

        float norm_l1;
        bool found;
        int found_id = 0;
        cv::Point2f temp_point;
        int total_duplicate_found = 0;

        for (int i=0; i<new_points.size(); ++i){
            found = false;

            for (tie(v,vend) = vertices(tracks_connection_graph_); v!=vend; ++v){
                temp_point = tracks_connection_graph_[*v].pos;

                norm_l1 = abs( temp_point.x - new_points[i].x +
                                temp_point.y - new_points[i].y );

                if ( norm_l1 < min_distance_between_tracks_){
                    found = true;
                    found_id = tracks_connection_graph_[*v].id;
                    total_duplicate_found++;
                    break;
                }
            }

            if (found)
                (*assigned_ids)[i] = found_id;
            else
                (*assigned_ids)[i] = -1;

        }
#ifdef DEBUG_PRINOUT
        // Debug: how many duplicates were that
        printf("Found duplicates: %d/%d\n", total_duplicate_found, new_points.size());
#endif
    }

    float TrackManager::Distance(const TracksConnectionGraph::vertex_descriptor & v1, const TracksConnectionGraph::vertex_descriptor & v2){
        // L2 distance between two tracks' positions

        const cv::Point2f point1 = tracks_connection_graph_[v1].pos;
        const cv::Point2f point2 = tracks_connection_graph_[v2].pos;

        return sqrt( (point1.x - point2.x) * (point1.x - point2.x) +
                     (point1.y - point2.y) * (point1.y - point2.y)
                   );
    }

    ConnectedComponents TrackManager::GetConnectedComponents() const{
        // Call Boost Graph connected component function
        typedef std::map<TracksConnectionGraph::vertex_descriptor, TracksConnectionGraph::vertices_size_type> component_type;
        component_type component;
        boost::associative_property_map< component_type > component_map(component);

        int num_components = connected_components(tracks_connection_graph_, component_map);

        ConnectedComponents return_connected_components(num_components);

        // This gives us a mapping
        // vertex_id => component_id

        // What we need now is
        // component_id => [vertex_id_1, vertex_id_2, vertex_id_3, ...]

        //assert(all_tracks.size() == component.size())
        TracksConnectionGraph::vertices_size_type component_id;
        component_type::iterator it = component.begin();
        for (; it != component.end(); it++){
            component_id = (*it).second; // this gives the id that this track is assigned to
            return_connected_components[component_id].push_back(
                    get_track_information((*it).first)
                    );
        }

        return return_connected_components;
    }

    void TrackManager::GetAllTracksPositionAndId(std::vector<cv::Point2f> * frame_points, std::vector<int> * ids){
        frame_points->clear();
        ids->clear();
        frame_points->resize(num_vertices(tracks_connection_graph_));
        ids->resize(num_vertices(tracks_connection_graph_));

        TracksConnectionGraph::vertex_iterator vi, viend;

        int i=0;
        for (tie(vi,viend) = vertices(tracks_connection_graph_); vi!=viend; ++vi){
            (*frame_points)[i] = tracks_connection_graph_[*vi].pos;
            (*ids)[i] = tracks_connection_graph_[*vi].id;
            i++;
        }
    }

    float TrackManager::Distance(const cv::Point2f &pt1, const cv::Point2f &pt2){
        return sqrt(
                    pow(pt1.x - pt2.x, 2) +
                    pow(pt1.y - pt2.y, 2)
                    );
    }

    /** \brief Given a list of ids, find the corresponding vertices

      Example: let's say we are interested in getting vertex descriptor for track id 10,12,25, we can do:

          interested_ids = [10,12,25]
          a = GetVertexDescriptors(interested_ids)

       Becareful though as these vertex descriptors will be invalidated if a vertex is removed

      \param old_points_ids IDs of the tracks you are interested in getting the vertices descriptors
      \return a vector of the same size of old_points_ids where each element point to the correponding vertex descriptor
    */
    std::vector<TracksConnectionGraph::vertex_descriptor> TrackManager::GetVertexDescriptors(std::vector<int> old_points_ids){
        // First construct a hash that store track_id => vertex_id
        std::map<int, TracksConnectionGraph::vertex_descriptor> tracks_vertices_map;

        // Go through all vertices and populate this map
        TracksConnectionGraph::vertex_iterator vi, viend;
        for (tie(vi,viend) = vertices(tracks_connection_graph_); vi!=viend; ++vi){
            tracks_vertices_map[ tracks_connection_graph_[*vi].id ] = *vi;
        }

        // Second, go through the ids we need to search and populate the results
        std::vector<TracksConnectionGraph::vertex_descriptor> found_vertex_descriptors(old_points_ids.size());

        for (int i=0; i<old_points_ids.size(); ++i){
            found_vertex_descriptors[i] = tracks_vertices_map[old_points_ids[i]];
        }

        return found_vertex_descriptors;
    }

    TrackInformation TrackManager::get_track_information(TracksConnectionGraph::vertex_descriptor v) const{
        TrackInformation track_info;
        track_info.pos = tracks_connection_graph_[v].pos;
        track_info.number_of_times_tracked = tracks_connection_graph_[v].number_of_times_tracked;
        track_info.id = tracks_connection_graph_[v].id;
        track_info.activated = tracks_connection_graph_[v].activated;
        track_info.last_time_tracked = tracks_connection_graph_[v].last_time_tracked;
        track_info.average_position = tracks_connection_graph_[v].average_position;

        return track_info;
    }

    void TrackManager::DeleteTrack(int id){
        TracksConnectionGraph::vertex_iterator vi, viend;
        tie(vi, viend) = vertices(tracks_connection_graph_);
        for (; vi!=viend; ++vi){
            if( tracks_connection_graph_[*vi].id == id ){
                clear_vertex(*vi, tracks_connection_graph_);
                remove_vertex(*vi, tracks_connection_graph_);
                return;
            }
        }
    }
}
