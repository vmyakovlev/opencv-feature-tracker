#include <opencv2/video/blobtrack2.hpp>
#include <opencv2/core/mat.hpp>

namespace cv{
    BlobWithTrajectoryMatcher::BlobWithTrajectoryMatcher(BlobTrajectoryTracker * trajectory_tracker){
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
    void BlobWithTrajectoryMatcher::match(const Mat &query_image, const std::vector<Blob> &query_blobs,
                                     const Mat &target_image, const std::vector<Blob> &target_blobs, std::vector<int> &matches){
        // clear the output variable
        matches.resize(query_blobs.size());

        for (int i=0; i<query_blobs.size(); i++){
            matches[i] = -1;

            // go through each target blob and find one that works
            for (int j=0; j<target_blobs.size(); j++){
                if (!isClose(query_blobs[i], target_blobs[i]))
                    continue;

                // found a close-by blob, time to check for trajectory
                if (!trajectory_tracker_->isTrajectoryConsistent(query_blobs[i], j)){
                    continue;
                }

                // we have found a match
                matches[i] = j;
            }
        }
    }

    /** \brief Tell whether a query blob is close to a target blob

      The query blob is close to a target blob when they "kind-of" overlap. This condition happens
      when the top left corner of the query is within a certain bounds of the top left corner of the
      target. This bound is defined as twice the size of the target blob.
    */
    bool BlobWithTrajectoryMatcher::isClose(const Blob & query, const Blob & target) const{
        cv::Rect query_bb = query.GetBoundingUprightRectangle();
        cv::Rect target_bb = target.GetBoundingUprightRectangle();

        float dx = fabs(query_bb.x - target_bb.x);
        float dy = fabs(query_bb.y - target_bb.y);

        return dx > 2*target_bb.width || dy > 2*target_bb.height;
    }


    BlobTrajectoryTracker::BlobTrajectoryTracker(){
        current_time_ = 0;
        next_blob_id_ = 0;
    }

    /** \brief Add new blobs to the tracker

      This method does not check whether these blobs are colliding with other blobs. i.e. These new blobs
      might be the same as those old blobs.
    */
    void BlobTrajectoryTracker::addTracks(const std::vector<Blob> & new_blobs){
        std::vector<Blob>::const_iterator it = new_blobs.begin();
        for (; it!=new_blobs.end(); it++){
            blobs_over_time_[current_time_][next_blob_id_] = *it;
            next_blob_id_++;
        }
    }

    /** \brief Update the blob information

      If you specify a track id that does not current exist, it will be added.

      \param tracks_to_update key is the id of the track, value is the new blob object to update

    */
    void BlobTrajectoryTracker::updateTracks(const std::map<int, Blob> & tracks_to_update){
        std::map<int, Blob>::const_iterator it = tracks_to_update.begin();
        for( ; it!=tracks_to_update.end(); it++){
            blobs_over_time_[current_time_][(*it).first] = (*it).second;
        }
    }

    /** \brief remove certain tracks from the tracker
      */
    void BlobTrajectoryTracker::removeTracks(const std::vector<int> & ids_to_remove){
        std::map<int, Blob>::iterator it;
        for (size_t i=0; i<ids_to_remove.size(); i++){
            it = blobs_over_time_[current_time_].find(ids_to_remove[i]);
            blobs_over_time_[current_time_].erase(it);
        }
    }

    /** \brief check if these new query blobs will fit into the trajectory

      A trajectory is consistent when the upcoming position conforms (within a certain range)
      with the velocity of the previous positions of this track.

      \param target_track_id ID of the track against which these new blobs are to be tested
      \param query_blob a blob to be considered for trajectory consistency
    */
    bool BlobTrajectoryTracker::isTrajectoryConsistent(const Blob & query_blob, int target_track_id) const {

    }

    /** \brief Advance to the next time instance
    */
    void BlobTrajectoryTracker::nextTimeInstance() {
        current_time_ ++;

        // we will need the new the blob information of this instance to start
        // the same as the first one
        blobs_over_time_.push_back(blobs_over_time_.at(blobs_over_time_.size()-1));
    }
}
