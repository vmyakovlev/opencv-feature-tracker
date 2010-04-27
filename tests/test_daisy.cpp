#include "daisy_feature.h"
#include "test_configuration.h"
#include <gtest/gtest.h>

#include <cv.h>
#include <highgui.h>
using cv::Mat;

TEST(TestDaisy, LoadImage){
    Mat im = imread(data_folder_path + "/frame_0000.jpg",0);

    ASSERT_EQ(720, im.cols);
    ASSERT_EQ(576, im.rows);
}

TEST(TestDaisy, ComputeDense){
    Mat im = imread(data_folder_path + "/frame_0000.jpg",0);

    DaisyDescriptorExtractor daisy_extractor;
    Mat descriptor;
    daisy_extractor.compute_dense(im, descriptor);

    ASSERT_EQ(720*576, descriptor.rows);
    ASSERT_EQ(200, descriptor.cols);
}

/** We want daisy to compute the descriptor in place instead of always
    allocating new memory.
*/
TEST(TestDaisy, ComputeInplace){
    Mat im = imread(data_folder_path + "/frame_0000.jpg",0);

    DaisyDescriptorExtractor daisy_extractor;
    Mat descriptor(720*576, 200, CV_32F);

    uchar * original_data = descriptor.data;

    daisy_extractor.compute_dense(im, descriptor);

    ASSERT_EQ(original_data, descriptor.data);
}
