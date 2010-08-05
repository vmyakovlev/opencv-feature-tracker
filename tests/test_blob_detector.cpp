#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include <opencv2/video/blobtrack2.hpp>
#include "test_configuration.h"
#include "Blob.h"

class BloBDetectorTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        simple_blob_image = cv::imread(data_folder_path + "/simple_21_blobs.png", 0);
        advanced_blob_image = cv::imread(data_folder_path + "/advanced_3_blobs.png", 0);
    }

    // virtual void TearDown() {}
    Mat simple_blob_image;
    Mat advanced_blob_image;
    cv::BlobDetector blob_detector;
};

TEST_F(BloBDetectorTest, SimpleBlobDetector){
    vector<Blob> blobs = blob_detector(simple_blob_image);

    ASSERT_EQ(21, blobs.size());
}

TEST_F(BloBDetectorTest, AdvancedBlobDetector){
    vector<Blob> blobs = blob_detector(advanced_blob_image);

    ASSERT_EQ(3, blobs.size());
}
