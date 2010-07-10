#ifndef __SAUNIERSAYED_FEATURE_GROUPING_
#define __SAUNIERSAYED_FEATURE_GROUPING_

#include <cv.h>
#include <map>
#include <vector>
#include <queue>
#include <boost/graph/adjacency_list.hpp>
#include <fstream>

using namespace boost;

/** Implemetation of Saunier Sayed 2006 algorithm. For the algorithm section, check page 4 of Saunier Sayed
  2006 A feature-based tracking algorithm for vehicles in intersections.
*/
namespace SaunierSayed{

    typedef struct LinkInformation_{
        int id; // id of this edge (currently not used/updated)
        double min_distance; // the minimum distance this track has ever had with the target track
        double max_distance; // the maximum distance this track has ever had with the target track
        bool active;
    } LinkInformation;

    typedef struct TrackInformation_{
        int id; // if of the track (vertex) (synced with vertex_index internally managed by BGL)
        cv::Point2f pos;
        std::deque<cv::Point2f> previous_points; // save a fixed set of previous points
        cv::Point2f average_position; // Average position of this track stored in previous_points. For tracks that have strong movements,
                                // their new positions are further and further away from the average position.
                                // For tracks that stay stationary (e.g. scene objects, parked vehicles), their
                                // new position stay around this value.
        std::deque<float> previous_displacements; // save a fixed set of previous displacements
                                                        // the number of previous displacements = number of previous points - 1
        int number_of_times_tracked;
        int last_time_tracked; // When was the last time-stamp that this track was found
        bool activated;
    } TrackInformation;

    typedef std::map<int, TrackInformation> Tracks;
    typedef std::vector<TrackInformation> ConnectedComponent;
    typedef std::vector<ConnectedComponent> ConnectedComponents;

    typedef adjacency_list <listS, vecS, undirectedS, TrackInformation, LinkInformation> TracksConnectionGraph;

    /** \class TrackManager

      This class implements a manager which group features based on how they move together. It implements the algorithm
      proposed in Saunier and Sayed 2006.

      \todo Systematically remove points that are not tracked
    */
    class TrackManager{
    public:
        /**
          The visualizer will need the graph information of this class in order to visualizer the graph on the image
          Thus either we make it a friend or we need to return the entire graph structure.
          Returning an entire graph structure means repackaging them into some data structures => too expensive
        */
        friend class FeatureGrouperVisualizer;

        TrackManager(int min_num_frame_tracked = 4,
                     float min_distance_moved_required = 3,
                     float maximum_distance_threshold = 20,
                     float feature_segmentation_threshold = 50,
                     float minimum_variance_required = 5,
                     float min_distance_between_tracks = 20,
                     bool log_track_to_file = false
                     );
        ~TrackManager();

        TrackManager(const TrackManager & other);
        TrackManager& operator=(const TrackManager & other);


        //! Add new tracks without checking if there is duplications
        /**
          One can pass in a pre-populated ids which can be generated from FindDuplicatePointIds() to the 2nd parameter.
          When this happens, if a specific element has already been found (i.e. assigned_ids[i] >=0), it will be skipped.

          \param[inout] assigned_ids The (newly) assigned ids for the input new_points

        */
        void AddPoints(const std::vector<cv::Point2f> & new_points, std::vector<int> * assigned_ids = NULL);

        //! Add newly detected points. Remove duplicates.
        /**
          \return The assigned ids for these points
         */
        void AddPossiblyDuplicatePoints(const std::vector<cv::Point2f> & new_points, std::vector<int> * assigned_ids);

        //! Remove points that is already in (i.e. distance to any existing track < epsilon) one of the tracks
        void RemoveDuplicatePoints(std::vector<cv::Point2f> & input_points);

        //! Find the Ids of the duplicate points
        void FindDuplicatePointIds(const std::vector<cv::Point2f> & new_points, std::vector<int> * assigned_ids = NULL);

        //! Update current tracks with new points
        void UpdatePoints(const std::vector<cv::Point2f> & new_points, const std::vector<int> & old_points_indices);

        //! Remove a track given its id
        void DeleteTrack(int id);

        //! Cut the link(edge) where max_distance - min_distance > D_segmentation
        void SegmentFarAwayTracks();

        //! Find the connected components in our graph structure
        /** The name is intentionally CamelCase despite methods that starts with get usually has lower cases.
          The reason is because this operation is not simple. The CamelCase signifies this.

          \return connected components (each connected component is a group of TrackInformation)
         */
        ConnectedComponents GetConnectedComponents() const;

        //! Return the number of tracks
        int num_tracks();

        //! Return the number of connections amongst tracks
        int num_connections();

        /** \brief Get current position of all tracks
        */
        void GetAllTracksPositionAndId(std::vector<cv::Point2f> * frame_points, std::vector<int> * ids);

        /** \brief Advance our feature grouper state to the next frame

          All point addition will be considered for the next time step instead.
          User should call this method when they are done with this time step
        */
        void AdvanceToNextFrame();

        //////// These API should be used with care since they are not fast due to data copying
        //! Return all tracks information
        Tracks tracks() const;

        //! Return edge information for a specfic vertex pairs
        /**
          Make sure you check the return value prior to accessing the data in output_link_information
          Since the graph structure is undirected. get_edge_information(1,2,&out) gives the same result as
          get_edge_information(2,1,&out)

          \param vertex_id_1 id of the first vertex in this edge
          \param vertex_id_2 id of the second vertex in this edge
          \param[out] output_link_information structure to write out the link information
          \return True if an edge exists
        */
        bool get_edge_information(int vertex_id_1, int vertex_id_2, LinkInformation * output_link_information) const;
    private:
        TrackInformation get_track_information(TracksConnectionGraph::vertex_descriptor v) const;

        //! Activate a track
        /** Activate a track will connect it to all tracks that are current close by
        */
        void ActivateTrack(TracksConnectionGraph::vertex_descriptor v);

        float Distance(const TracksConnectionGraph::vertex_descriptor & v1, const TracksConnectionGraph::vertex_descriptor & v2);
        static float Distance(const cv::Point2f & pt1, const cv::Point2f & pt2);

        /** \brief Log the current track information into internal log file

          Currently only log track positions
        */
        void LogCurrentTrackInfo();

        //! Given a list of ids, find the corresponding vertices
        std::vector<TracksConnectionGraph::vertex_descriptor> GetVertexDescriptors(std::vector<int> old_points_ids);

        // Methods used in UpdatePoints()
        void UpdatePoint(const cv::Point2f & new_position, const TracksConnectionGraph::vertex_descriptor & v);
        void UpdatePointMinMaxEdgeDistance(const TracksConnectionGraph::vertex_descriptor & v);
        void FindVerticesNotTrackedFor(int number_of_frames_not_tracked, std::set<int> * output_found_track_ids) const;
        void ActivateTracksTrackedLongEnough(const std::vector<TracksConnectionGraph::vertex_descriptor> & vertices_to_consider);
        void FindVerticesNotMovingEnough(float min_distance_moved_required,
                                                       int num_previous_points_to_check,
                                                       const std::vector<TracksConnectionGraph::vertex_descriptor> & vertices_to_consider,
                                                       std::set<int> *output_found_track_ids) const;

        TracksConnectionGraph tracks_connection_graph_;

        // Some parameters
        int min_num_frame_tracked_; //!< How many frames do we track this point until it is activated
        float min_distance_moved_required_;
        float maximum_distance_threshold_;
        float feature_segmentation_threshold_;
        int maximum_previous_points_remembered_;

        /** Tracks that are close to each other (L1 distance between them smaller than this value) is considered the same and will not be added
        This distance is in world coordinate (NOT in pixel coordinate) */
        float min_distance_between_tracks_;

        float minimum_variance_required_; //!< Minimum variance of motion in the last N frames. Tracks having less than this variance in motion will be removed.
        int max_num_frames_not_tracked_allowed_;

        // Some flags
        bool logging_; //!< Are we logging track information?

        // Variables for logging
        int current_time_stamp_id_;
        int next_track_id_;
        std::ofstream log_file_;
    };
}

#endif
