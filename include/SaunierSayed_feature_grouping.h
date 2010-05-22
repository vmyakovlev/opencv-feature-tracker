#ifndef __SAUNIERSAYED_FEATURE_GROUPING_
#define __SAUNIERSAYED_FEATURE_GROUPING_

#include <cv.h>
#include <map>
#include <vector>

namespace SaunierSayed{

    typedef struct LinkInformation_{
        double min_distance; // the minimum distance this track has ever had with the target track
        double max_distance; // the maximum distance this track has ever had with the target track
    } LinkInformation;

    typedef struct TrackInformation_{
        cv::Point2f pos;
        int number_of_times_tracked;
        bool activated;
        std::map<int, LinkInformation> links;
    } TrackInformation;

    typedef std::map<int, TrackInformation> Tracks;

    class TrackManager{
    public:
        TrackManager();

        //! Add new tracks
        void AddPoints(const std::vector<cv::Point2f> & new_points);

        //! Update current tracks with new points
        void UpdatePoints(const std::vector<cv::Point2f> & new_points, const std::vector<int> & old_points_indices);

        Tracks & tracks();
    private:
        //! The id to be used for the next new track
        int next_id_;
        Tracks tracks_;
    };
}

#endif
