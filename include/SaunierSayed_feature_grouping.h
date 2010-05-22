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
        cv::Point position;
        int number_of_times_tracked;
        bool activated;
        std::map<int, LinkInformation> links;
    } TrackInformation;

    class TrackManager{
    public:
        void AddPoints(const std::vector<cv::Point2f> new_points);
        void UpdatePoints(const std::vector<cv::Point2f> & new_points, const std::vector<int> & old_points_indices);

        std::map<int, TrackInformation> tracks;
    };
}

#endif
