#include <opencv2/video/blobtrack2.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "misc.h"

namespace cv {
    /** Save a pointer to the trajectory tracker so that whenever you are asked to match, we can
      use the trajectory tracker to determine whether a given blob will fit into the trajectory
      model.

      \todo Remove dependency to BlobTrajectoryTracker, instead use its superclass
    */
    BlobMatcherWithTrajectory::BlobMatcherWithTrajectory(BlobTrajectoryTracker * trajectory_tracker){
        trajectory_tracker_ = trajectory_tracker;
    }

    /** Match input/query blobs to target blobs

      This matcher tries to match a query blob by choosing amongst the target blobs:
      * the closest blob
      * the blob that has similar trajectory

      For the original implementation that inspired this implementation, please see
      enteringblobdetection.cpp line 817-963 of OpenCV2 SVN r3399

      \param query_image the image where the blobs were found (i.e. from BlobDetector)
      \param query_blobs the results from a BlobDetector
      \param target_image the image where the target_blobs were found
      \param target_blobs the results from a BlobDetector for target_image
      \param[out] matches the mapping which tells what each blob in query_blobs is matched to

      \todo Handle the case where there are two close-by blobs
      \todo Remove blobs that are too close to the borders
    */
    void BlobMatcherWithTrajectory::match(const Mat & query_image, const std::vector<Blob> & query_blobs,
                                                       std::vector<BlobTracker::id_type> & matches) const{
        // clear the output variable
        matches.resize(query_blobs.size());

        float best_error = 1e11;
        BlobTracker::id_type best_track_id = -1;
        float current_error;
        std::map<BlobTracker::id_type, Blob> target_blobs = trajectory_tracker_->getBlobs();
        std::map<BlobTracker::id_type, Blob>::const_iterator it = target_blobs.begin();

        /* Debug: Drawing the query and target blobs
           on the window
        */
        const std::string query_blob_window = "Query Blob Window";
        const std::string target_blob_window = "Target Blob Window";

        cv::namedWindow(query_blob_window);
        cv::namedWindow(target_blob_window);

        cv::Size size(720, 576);
        cv::Mat query_blob_img = cv::Mat::zeros(size, CV_8UC3);
        cv::Mat target_blob_img = cv::Mat::zeros(size, CV_8UC3);

        for (int i=0; i<query_blobs.size(); i++){
            query_blobs[i].DrawTo(query_blob_img, "", CV_RGB(255,255,0));
            std::cout << ".";
        }
        for (; it!=target_blobs.end(); it++){
            (*it).second.DrawTo(query_blob_img, "+", CV_RGB(255,0,255));
            std::cout << "+";
        }

        cv::imshow(query_blob_window, query_blob_img);
        cv::imshow(target_blob_window, target_blob_img);
        cv::waitKey(0);

        /* End debug drawing */

        for (int i=0; i<query_blobs.size(); i++){
            matches[i] = -1;

            // go through each target blob and find one that works
            for (it = target_blobs.begin(); it!=target_blobs.end(); it++){
                if (!isClose(query_blobs[i], (*it).second))
                    continue;
                std::cout << "*";

                // found a close-by blob, time to check for trajectory
                if (!trajectory_tracker_->isTrajectoryConsistent(query_blobs[i], (*it).first, current_error)){
                    continue;
                }
                std::cout << "^";

                // we have found a match
                matches[i] = (*it).first;
                std::cout << "Found a match " << (*it).first << " for " << i << std::endl;
                break;
            }
        }

        print(matches);
    }

    /** \brief Tell whether a query blob is close to a target blob

      The query blob is close to a target blob when they overlap.
    */
    bool BlobMatcherWithTrajectory::isClose(const Blob & query, const Blob & target) const{
        cv::Rect query_bb = query.GetBoundingUprightRectangle();
        cv::Rect target_bb = target.GetBoundingUprightRectangle();

        // We find the four conditions in which the two bounding boxes do not overlap
        // and invert it
        return !(query_bb.x > target_bb.x + target_bb.width ||
                query_bb.x + query_bb.width < target_bb.x ||
                query_bb.y > target_bb.y + target_bb.height ||
                query_bb.y + query_bb.height < target_bb.y
                );
    }


    BlobTrajectoryTracker::BlobTrajectoryTracker(){
        current_time_ = 0;
        next_blob_id_ = 0;

        // initialize internal storage so we can start assigning right away
        blobs_over_time_.push_back(std::map<BlobTracker::id_type, Blob>());
    }

    /** \brief Add new blobs to the tracker

      This method does not check whether these blobs are colliding with other blobs. i.e. These new blobs
      might be the same as those old blobs.
    */
    void BlobTrajectoryTracker::addTracks(const std::vector<Blob> & new_blobs){
        std::vector<Blob>::const_iterator it = new_blobs.begin();
        for (; it!=new_blobs.end(); it++){
            addTrack(*it);
        }
    }

    /** \brief Add a track to the tracker
    */
    void BlobTrajectoryTracker::addTrack(const Blob & new_blob){
        blobs_over_time_[current_time_][next_blob_id_] = new_blob;

        TrackedObjectInformation track_info;
        track_info.first_tracked_time_stamp = current_time_;
        track_info.last_tracked_time_stamp = -1;
        track_info.active = true;

        tracks_information[next_blob_id_] = track_info;

        next_blob_id_++;
    }

    /** \brief Update the blob information

      If you specify a track id that does not current exist, it will be added unless the track id is less than 0.
      This is a typical scenario when one uses a matcher. The matcher typically assigns ids of the found
      matches. For unmatched queries, a value less than 0 (e.g. -1) is assigned by the matcher.

      \param tracks_to_update key is the id of the track, value is the new blob object to update
      \param is_unmatched_will_get_created Default to false. If a match is <0, it will be created instead of ignored. For ids that
                                           do not exists, they are still ignored.

    */
    void BlobTrajectoryTracker::updateTracks(const std::map<BlobTracker::id_type, Blob> & tracks_to_update, bool is_unmatched_will_get_created /* = false */){
        std::map<BlobTracker::id_type, Blob>::const_iterator it = tracks_to_update.begin();
        std::map<BlobTracker::id_type, Blob>::iterator it2;

        for( ; it!=tracks_to_update.end(); it++){
            // is this a request for a new element
            if (is_unmatched_will_get_created && (*it).first == -1){
                addTrack((*it).second);
                continue;
            }

            // can we find this id?
            it2 = blobs_over_time_[current_time_].find((*it).first);

            if (it2 == blobs_over_time_[current_time_].end()) // cannot find this id
                continue;
            else {
                (*it2).second = (*it).second;
                tracks_information[(*it).first].last_tracked_time_stamp = current_time_;
            }
        }
    }

    /** \brief Remove certain tracks from the tracker
      */
    void BlobTrajectoryTracker::removeTracks(const std::vector<BlobTracker::id_type> & ids_to_remove){
        std::map<BlobTracker::id_type, Blob>::iterator it;
        for (size_t i=0; i<ids_to_remove.size(); i++){
            it = blobs_over_time_[current_time_].find(ids_to_remove[i]);
            if (it != blobs_over_time_[current_time_].end())
                blobs_over_time_[current_time_].erase(it);
        }
    }

    /** \brief Check if these new query blobs will fit into the trajectory

      A trajectory is consistent when the upcoming position conforms (within a certain range)
      with the velocity of the previous positions of this track.

      We will estimate the parameters of moving x(t) = a*t + b . The technique to perform linear
      fitting is Least Square Fitting: http://mathworld.wolfram.com/LeastSquaresFitting.html

      For a previous implementation, please check enteringblobdetection.cpp 895-939

      \param target_track_id ID of the track against which these new blobs are to be tested
      \param query_blob a blob to be considered for trajectory consistency
      \param[out] the error incurred when assigning this blob into this track
    */
    bool BlobTrajectoryTracker::isTrajectoryConsistent(const Blob & query_blob, BlobTracker::id_type target_track_id, float & error) const {
        int track_size = current_time_ - 1; // since we don't throw away old information, the size of a track
                                            // is the same as the previous timestamp
                                            // TODO: Check that this is still a valid assumption
        float       sum[2] = {0,0};
        float       jsum[2] = {0,0};
        float       a[2],b[2]; /* estimated parameters of moving x(t) = a*t+b*/

        std::vector<int> time_instances_where_track_exists;

        std::map<BlobTracker::id_type, Blob>::const_iterator it;
        for(int j=0; j<track_size; ++j)
        {
            it = blobs_over_time_[j].find(target_track_id);
            if (it == blobs_over_time_[j].end())
                continue;

            cv::Point2f blob_center = (*it).second.GetBoundingRectangle().center;
            time_instances_where_track_exists.push_back(j);

            sum[0] += blob_center.x;
            jsum[0] += j * blob_center.x;
            sum[1] += blob_center.y;
            jsum[1] += j * blob_center.y;
        }

        if (time_instances_where_track_exists.size() <= 5){
            // we don't have enough information about this track to reject it
            error = 1e10;
            return true;
        }

        a[0] = 6*((1-track_size)*sum[0]+2*jsum[0])/(track_size*(track_size*track_size-1));
        b[0] = -2*((1-2*track_size)*sum[0]+3*jsum[0])/(track_size*(track_size+1));
        a[1] = 6*((1-track_size)*sum[1]+2*jsum[1])/(track_size*(track_size*track_size-1));
        b[1] = -2*((1-2*track_size)*sum[1]+3*jsum[1])/(track_size*(track_size+1));

        // We will need to now calculate standard deviation of the errors
        float E_x = 0; // E(x)
        float E_x2 = 0; // E(x^2)
        for (int i=0; i<time_instances_where_track_exists.size(); i++){
            it = blobs_over_time_[i].find(target_track_id);
            cv::Point2f blob_center = (*it).second.GetBoundingRectangle().center;
            float this_point_error = pow(blob_center.x - (a[0] * i + b[0]), 2) +
                    pow(blob_center.y - (a[1] * i + b[1]), 2);

            E_x = E_x + this_point_error;
            E_x2 = E_x2 = pow(this_point_error, 2);
        }
        E_x = E_x / time_instances_where_track_exists.size();
        E_x2 = 1.0 / time_instances_where_track_exists.size() * E_x2;
        float std_deviation = sqrt(E_x2 + pow(E_x,2)); // the typical way to calculate std deviation

        // now that we have the estimates of a and b for both x(t) and y(t)
        // let's test that our input blob fits into this estimation
        cv::Point2f query_blob_center = query_blob.GetBoundingRectangle().center;
        error = pow(query_blob_center.x - (a[0] * current_time_ + b[0]), 2) +
                pow(query_blob_center.y - (a[1] * current_time_ + b[1]), 2);

        // if the error is within two standard deviation of the previous errors, accept
        float error_z = (error - E_x) / std_deviation;
        if (error_z < -2 || error_z > 2)
            return false;

        // this blob fits into this track
        return true;
    }

    /** \brief Advance to the next time instance
    */
    void BlobTrajectoryTracker::nextTimeInstance() {
        current_time_ ++;

        // we will need the new the blob information of this instance to start
        // the same as the last one
        std::map<BlobTracker::id_type, Blob> blobs_in_last_time_instance = blobs_over_time_.at(blobs_over_time_.size()-1);
        blobs_over_time_.push_back(blobs_in_last_time_instance);
    }

    int BlobTrajectoryTracker::numTracks() const{
        if (blobs_over_time_.size() == 0)
            return 0;

        return blobs_over_time_[current_time_].size();
    }

    /** \brief Get all blobs in the given time instance

      \param time_stamp the time instance we are interested in. If -1, return the current one.
    */
    std::map<BlobTracker::id_type, Blob> BlobTrajectoryTracker::getBlobs(int time_stamp /* = -1 */) const {
        if (time_stamp == -1)
            return blobs_over_time_[current_time_];
        else if (time_stamp <= current_time_)
            return blobs_over_time_[time_stamp];
        else
            throw std::range_error("Given time stamp is out of range");
    }

    TrackedObjectInformation BlobTrajectoryTracker::getTrackInformation(BlobTracker::id_type id) const {
        std::map<BlobTracker::id_type, TrackedObjectInformation>::const_iterator it;
        it = tracks_information.find(id);
        if (it != tracks_information.end()){
            return (*it).second;
        } else {
            TrackedObjectInformation bad_info;
            bad_info.active = false;
            bad_info.first_tracked_time_stamp = -1;
            bad_info.last_tracked_time_stamp = -1;
            return bad_info;
        }
    }
}
