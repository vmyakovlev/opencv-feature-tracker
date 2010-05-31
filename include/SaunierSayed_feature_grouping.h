#ifndef __SAUNIERSAYED_FEATURE_GROUPING_
#define __SAUNIERSAYED_FEATURE_GROUPING_

#include <cv.h>
#include <map>
#include <vector>
#include <boost/graph/adjacency_list.hpp>
using namespace boost;

/** Implemetation of Saunier Sayed 2006 algorithm. For the algorithm section, check page 4 of Saunier Sayed
  2006 A feature-based tracking algorithm for vehicles in intersections.
*/
namespace SaunierSayed{

    typedef struct LinkInformation_{
        int id; // id of this edge (currently not used/updated)
        double min_distance; // the minimum distance this track has ever had with the target track
        double max_distance; // the maximum distance this track has ever had with the target track
    } LinkInformation;

    typedef struct TrackInformation_{
        int id; // if of the track (vertex) (synced with vertex_index internally managed by BGL)
        cv::Point2f pos;
        int number_of_times_tracked;
        bool activated;
    } TrackInformation;

    typedef std::map<int, TrackInformation> Tracks;
    typedef std::vector<TrackInformation> ConnectedComponent;
    typedef std::vector<ConnectedComponent> ConnectedComponents;

    typedef adjacency_list <listS, vecS, undirectedS, TrackInformation, LinkInformation> TracksConnectionGraph;

    class TrackManager{
    public:
        TrackManager(int min_num_frame_tracked = 2, float maximum_distance_threshold = 4, float feature_segmentation_threshold = 25);

        //! Add new tracks without checking if there is duplications
        void AddPoints(const std::vector<cv::Point2f> & new_points);

        //! Add newly detected points. Remove duplicates.
        void AddPossiblyDuplicatePoints(const std::vector<cv::Point2f> & new_points);

        //! Remove points that is already in one of the tracks
        void RemoveDuplicatePoints(std::vector<cv::Point2f> & input_points);

        //! Update current tracks with new points
        void UpdatePoints(const std::vector<cv::Point2f> & new_points, const std::vector<int> & old_points_indices);

        //! Activate a track
        /** Activate a track will connect it to all tracks that are current close by
        */
        void ActivateTrack(int id);

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

        //////// These API should be used with care since they are not fast
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
        float Distance(const TracksConnectionGraph::vertex_descriptor & v1, const TracksConnectionGraph::vertex_descriptor & v2);
        TracksConnectionGraph tracks_connection_graph_;

        // Some parameters
        int min_num_frame_tracked_;
        float maximum_distance_threshold_;
        float feature_segmentation_threshold_;
    };
}

#endif
