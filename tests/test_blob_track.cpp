#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include <opencv2/video/blobtrack2.hpp>
#include "test_configuration.h"
#include "Blob.h"

namespace cv {
    class BlobTrackTest : public ::testing::Test {
    protected:
        BlobTrackTest():
                tracker(),
                matcher(&tracker){
        }

        virtual void SetUp() {
            four_blank_blobs.push_back(Blob());
            four_blank_blobs.push_back(Blob());
            four_blank_blobs.push_back(Blob());
            four_blank_blobs.push_back(Blob());
        }

        std::vector<Blob> four_blank_blobs;
        cv::BlobTrajectoryTracker tracker;
        cv::BlobMatcherWithTrajectory matcher;
    };

    TEST_F(BlobTrackTest, AddNewTracks){
        int num_tracks = tracker.numTracks();

        ASSERT_EQ(0, num_tracks);

        tracker.addTracks(four_blank_blobs);

        ASSERT_EQ(num_tracks+4, tracker.numTracks());

        std::map<BlobTracker::id_type, Blob> all_blobs = tracker.getBlobs();

        ASSERT_EQ(4, all_blobs.size());
    }

    TEST_F(BlobTrackTest, UpdateTracks){
        int num_tracks = tracker.numTracks();
        ASSERT_EQ(0, num_tracks);

        tracker.addTracks(four_blank_blobs);

        std::map<BlobTracker::id_type, Blob> updates;
        updates[0] = Blob();
        updates[1] = Blob();
        updates[2] = Blob();
        updates[3] = Blob();
        updates[-1] = Blob(); // this one got assigned -1, will get created when 2nd parameter is true
        updates[53] = Blob(); // this one doesn't exists, will always get ignored

        tracker.updateTracks(updates);

        ASSERT_EQ(4, tracker.numTracks());

        // Test that if we uses -1 as id, that blob will get created
        tracker.updateTracks(updates, true);
        ASSERT_EQ(5, tracker.numTracks());
    }

    TEST_F(BlobTrackTest, RemoveTracks){
        int num_tracks = tracker.numTracks();
        ASSERT_EQ(0, num_tracks);

        tracker.addTracks(four_blank_blobs);

        std::vector<BlobTracker::id_type> ids_to_remove;
        ids_to_remove.push_back(0);
        ids_to_remove.push_back(-1);
        ids_to_remove.push_back(2);
        ids_to_remove.push_back(3);
        ids_to_remove.push_back(4);
        ids_to_remove.push_back(5);
        ids_to_remove.push_back(6);

        tracker.removeTracks(ids_to_remove);

        // only track id 1 is left now
        // also notice that ids that do not exists are ignored
        ASSERT_EQ(1, tracker.numTracks());
    }

    TEST_F(BlobTrackTest, AdvanceTracker){
        int num_tracks = tracker.numTracks();
        ASSERT_EQ(0, num_tracks);

        tracker.addTracks(four_blank_blobs);
        num_tracks = tracker.numTracks();

        tracker.nextTimeInstance();

        ASSERT_EQ(num_tracks, tracker.numTracks());
    }

    TEST_F(BlobTrackTest, TrajectoryReturnsTrueForFirstCoupleOfFrames){
        int num_tracks = tracker.numTracks();

        tracker.addTracks(four_blank_blobs);
        num_tracks = tracker.numTracks();

        tracker.nextTimeInstance();

        float error;
        ASSERT_EQ(true, tracker.isTrajectoryConsistent(Blob(), 0, error));
        tracker.nextTimeInstance();
        ASSERT_EQ(true, tracker.isTrajectoryConsistent(Blob(), 0, error));
        tracker.nextTimeInstance();
        ASSERT_EQ(true, tracker.isTrajectoryConsistent(Blob(), 0, error));
    }

    TEST_F(BlobTrackTest, TrajectoryIsClose){
        std::vector<cv::Point2f> blob_points_1;
        blob_points_1.push_back(cv::Point2f(0,0));
        blob_points_1.push_back(cv::Point2f(0,100));
        blob_points_1.push_back(cv::Point2f(100,100));
        blob_points_1.push_back(cv::Point2f(100,0));
        Blob blob1(blob_points_1);

        std::vector<cv::Point2f> blob_points_2;
        blob_points_2.push_back(cv::Point2f(0,0));
        blob_points_2.push_back(cv::Point2f(0,50));
        blob_points_2.push_back(cv::Point2f(50,50));
        blob_points_2.push_back(cv::Point2f(50,0));
        Blob blob2(blob_points_2);

        std::vector<cv::Point2f> blob_points_3;
        blob_points_3.push_back(cv::Point2f(60,60));
        blob_points_3.push_back(cv::Point2f(60,100));
        blob_points_3.push_back(cv::Point2f(100,100));
        blob_points_3.push_back(cv::Point2f(100,60));
        Blob blob3(blob_points_3);

        // These blobs overlap quite a bit, make sure that our isClose function agrees
        ASSERT_EQ(true, matcher.isClose(blob1, blob2));
        ASSERT_EQ(true, matcher.isClose(blob1, blob3));
        ASSERT_EQ(false, matcher.isClose(blob2, blob3));
    }
}
