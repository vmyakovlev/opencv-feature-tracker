#ifndef __BLOB_DETECTOR_
#define __BLOB_DETECTOR_

#include "common.hpp"
#include "Blob.h"
#include <vector>
#include <stdexcept>

namespace cv{
    /** forward declarations to save on compile time*/
    class Mat;


    class BlobDetector{
    public:
        BlobDetector();
        //BlobDetector(); // fully parameterized specification
        virtual std::vector<Blob> operator()(const Mat & input_foreground_mask_image, int close_holes = 1) const;
    private:
        DISALLOW_COPY_AND_ASSIGN(BlobDetector);
    };


    class BlobMatcher {
    public:
        /** \brief interface for matching a set of query blobs (i.e. blobs detected in the current frames) against
          a set of target blobs (i.e. blobs detected in the previous frames).

          Subclass this class in order to implement your own blob matcher. Try not to store any information in your blob
          matcher implementation. If you are interested in implementing a more sophisticated blob matching technique (e.g.
          Kalman-filter technique, trajectory-modelling technique), you should implement their logic inside of a subclass
          of BlobTracker or perhaps as a general tracker.

          You will need to implement the match method which matches some input query_image and query_blobs
          If your matching method requires 2 inputs (query and target), you will implement another method that
          takes in the target and implement match for matching.

          This way, our matcher interface applies even when the matcher doesn't keep track of the possible target blobs.
        */
        virtual void match(const Mat & query_image, const std::vector<Blob> & query_blobs,
                           std::vector<int> & matches) = 0;
    };


    class BlobTrajectoryTracker {
    public:
        BlobTrajectoryTracker();
        void addTracks(const std::vector<Blob> & new_blobs);
        void updateTracks(const std::map<int, Blob> & tracks_to_update);
        void removeTracks(const std::vector<int> & ids_to_remove);
        bool isTrajectoryConsistent(const Blob & query_blobs, int target_track_id, float & error) const;
        void nextTimeInstance();

        int numTracks() const;
        std::map<int, Blob> getBlobs(int time_stamp = -1) const;
    private:
        std::vector<std::map<int, Blob> > blobs_over_time_;
        int current_time_;
        int next_blob_id_;
    };


    /** \class BlobMatcherWithTrajectory

      \todo Constructor should only need const version of BlobTrajectoryTracker
     */
    class BlobMatcherWithTrajectory : public BlobMatcher {
    public:
        BlobMatcherWithTrajectory(BlobTrajectoryTracker * trajectory_tracker);
        virtual void match(const Mat & query_image, const std::vector<Blob> & query_blobs,
                           std::vector<int> & matches);
    private:
        DISALLOW_COPY_AND_ASSIGN(BlobMatcherWithTrajectory);

        bool isClose(const Blob & query, const Blob & target) const;

        BlobTrajectoryTracker * trajectory_tracker_;
    };
}

#endif
