#include "SaunierSayed_feature_grouping.h"

namespace SaunierSayed{
    TrackManager::TrackManager(int min_num_frame_tracked, float maximum_distance_threshold, float feature_segmentation_threshold){
        next_id_ = 0;
        min_num_frame_tracked_ = min_num_frame_tracked;
        maximum_distance_threshold_ = maximum_distance_threshold;
        feature_segmentation_threshold_ = feature_segmentation_threshold;
    }

    void TrackManager::AddPoints(const std::vector<cv::Point2f> & new_points){
        for (int i=0; i<new_points.size(); ++i){
            tracks_[next_id_].pos =  new_points[i];
            tracks_[next_id_].activated = false;
            tracks_[next_id_].number_of_times_tracked = 1;
            next_id_++;
        }
    }

    void TrackManager::UpdatePoints(const std::vector<cv::Point2f> & new_points, const std::vector<int> & old_points_indices){
        for (int i=0; i<old_points_indices.size(); i++){
            tracks_[old_points_indices[i]].pos = new_points[i];
            tracks_[old_points_indices[i]].number_of_times_tracked++;

            // NOTE: activation only happens once
            if (tracks_[old_points_indices[i]].number_of_times_tracked == min_num_frame_tracked_){
                ActivateTrack(i);
            }
        }
    }

    void TrackManager::ActivateTrack(int id){
        tracks_[id].activated = true;

        // find all close-by tracks and connect them
    }

    Tracks & TrackManager::tracks(){
        return tracks_;
    }

    void TrackManager::AddPossiblyDuplicatePoints(const std::vector<cv::Point2f> & new_points){
        std::vector<cv::Point2f> copy_points = new_points;
        RemoveDuplicatePoints(copy_points);
        AddPoints(copy_points);
    }

    //! Remove points that is already in one of the tracks
    void TrackManager::RemoveDuplicatePoints(std::vector<cv::Point2f> & input_points){
        std::vector<cv::Point2f> cleaned_input_points;

        // if the point positions are indexed, this search could be a bit better
        TracksIterator it;
        std::vector<cv::Point2f>::iterator input_point;

        float norm_l1;
        bool to_be_removed;
        cv::Point2f temp_point;
        for (input_point=input_points.begin(); input_point!=input_points.end(); ++input_point){
            to_be_removed = false;

            for (it=tracks_.begin(); it!=tracks_.end(); ++it){
                temp_point = (*it).second.pos;
                norm_l1 = abs( temp_point.x - (*input_point).x +
                                temp_point.y - (*input_point).y );

                if ( norm_l1 < 2){
                    to_be_removed = true;
                    break;
                }
            }

            if (!to_be_removed)
                cleaned_input_points.push_back(*input_point);
        }

        input_points = cleaned_input_points;
    }
}
