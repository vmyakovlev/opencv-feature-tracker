#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include <opencv2/video/blobtrack2.hpp>
#include "test_configuration.h"
#include "Blob.h"

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
}

TEST_F(BlobTrackTest, UpdateTracks){
    int num_tracks = tracker.numTracks();
    ASSERT_EQ(0, num_tracks);

    tracker.addTracks(four_blank_blobs);

    std::map<int, Blob> updates;
    updates[0] = Blob();
    updates[1] = Blob();
    updates[2] = Blob();
    updates[3] = Blob();
    updates[-1] = Blob();

    ASSERT_EQ(5, updates.size());

    tracker.updateTracks(updates);

    ASSERT_EQ(4, tracker.numTracks());
}
