#include "SaunierSayed_feature_grouping.h"

namespace SaunierSayed{
    TrackManager::TrackManager(){
        next_id_ = 0;
    }

    void TrackManager::AddPoints(const std::vector<cv::Point2f> & new_points){
        for (int i=0; i<new_points.size(); ++i){
            tracks_[next_id_].pos =  new_points[i];
            next_id_++;
        }

    }

    void TrackManager::UpdatePoints(const std::vector<cv::Point2f> & new_points, const std::vector<int> & old_points_indices){
        for (int i=0; i<old_points_indices.size(); i++){
            tracks_[old_points_indices[i]].pos = new_points[i];
        }
    }

    Tracks & TrackManager::tracks(){
        return tracks_;
    }
}
